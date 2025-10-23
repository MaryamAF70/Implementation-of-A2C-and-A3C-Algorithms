import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import time


# -------------------------
# Actor-Critic Network
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        value = self.critic(x)
        return mean, std, value


# -------------------------
# Worker Process
# -------------------------
def worker_process(global_model, grad_queue, log_queue, args, process_id, episode_counter):
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    local_model = ActorCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim)
    local_model.load_state_dict(global_model.state_dict())

    episodes_per_proc = args.max_episodes // args.num_processes
    all_rewards = []

    for _ in range(episodes_per_proc):
        state, _ = env.reset()
        log_probs, values, rewards = [], [], []
        done = False
        steps = 0

        while not done and steps < 200:  # جلوگیری از لوپ بی‌نهایت
            steps += 1
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std, value = local_model(state_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.clamp(action, -action_bound, action_bound)

            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy()[0])
            done = terminated or truncated

            rewards.append(torch.tensor([reward], dtype=torch.float32))
            values.append(value)
            log_probs.append(dist.log_prob(action).sum(dim=-1, keepdim=True))
            state = next_state

        # محاسبه Returns
        returns = []
        G = torch.zeros(1)
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        advantage = returns - values.squeeze()

        # Loss
        actor_loss = -(log_probs.squeeze() * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        # محاسبه گرادیان روی local
        local_model.zero_grad()
        loss.backward()

        grads = [p.grad.clone() for p in local_model.parameters() if p.grad is not None]
        grad_queue.put(grads)

        # شمارنده اپیزود
        with episode_counter.get_lock():
            episode_counter.value += 1
            global_ep = episode_counter.value

        ep_reward = float(sum([r.item() for r in rewards]))
        all_rewards.append(ep_reward)

        if global_ep % args.log_interval == 0:
            recent_rewards = all_rewards[-args.log_interval:]
            mean_recent = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
            log_queue.put((global_ep, ep_reward, mean_recent))

    env.close()
    log_queue.put("DONE")


# -------------------------
# A3C Training Function
# -------------------------
def train_a3c(args):
    mp.set_start_method("spawn", force=True)

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    global_model = ActorCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=args.lr)

    grad_queue = mp.Queue()
    log_queue = mp.Queue()
    episode_counter = mp.Value('i', 0)

    start_time = time.time()
    processes = []
    for i in range(args.num_processes):
        p = mp.Process(
            target=worker_process,
            args=(global_model, grad_queue, log_queue, args, i, episode_counter)
        )
        p.start()
        processes.append(p)

    finished_processes = 0
    all_rewards = []

    # حلقه‌ی مرکزی: آپدیت مدل + مانیتور لاگ‌ها
    while finished_processes < args.num_processes:
        try:
            msg = log_queue.get(timeout=5)
        except:
            msg = None

        if msg == "DONE":
            finished_processes += 1
        elif msg is not None:
            global_ep, ep_reward, mean_recent = msg
            print(f"[A3C] Episode {global_ep} | Reward: {ep_reward:.2f} | MeanReward({args.log_interval}): {mean_recent:.2f}")
            all_rewards.append(ep_reward)

        # گرادیان‌ها رو از صف بگیر و روی مدل global اعمال کن
        while not grad_queue.empty():
            grads = grad_queue.get()
            optimizer.zero_grad()
            for param, grad in zip(global_model.parameters(), grads):
                param.grad = grad
            optimizer.step()

    for p in processes:
        p.join()

    elapsed_time = time.time() - start_time
    return all_rewards, elapsed_time