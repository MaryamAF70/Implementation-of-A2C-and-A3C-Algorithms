# train_a3c_a2c_continuous.py
# اصلاح‌شده برای Gymnasium + PyTorch
# پشتیبانی از NUM_WORKERS قابل تنظیم

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np

# هایپرپارامترها
MAX_EPISODE = 50000
GAMMA = 0.99
ENTROPY_BETA = 0.05
PRINT_EVERY = 1000
LR = 5e-5
UPDATE_GLOBAL_ITER = 5
NUM_WORKERS = 4  

env_name = "Pendulum-v1"


# ---------------- شبکه Actor-Critic ---------------- #
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 100)

        # Actor
        self.mu = nn.Linear(100, a_dim)
        self.sigma = nn.Linear(100, a_dim)

        # Critic
        self.v = nn.Linear(100, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # اضافه کردن batch dim
        x = F.relu(self.fc1(x))
        mu = 2 * torch.tanh(self.mu(x))  # اکشن Pendulum در [-2, 2]
        sigma = F.softplus(self.sigma(x)) + 1e-5
        values = self.v(x)
        return mu, sigma, values


# ---------------- Worker Process ---------------- #
def worker(global_net, optimizer, global_ep, res_queue, idx):
    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    local_net = Net(s_dim, a_dim)
    total_step = 1

    while global_ep.value < MAX_EPISODE:
        s, _ = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_reward = 0

        for t in range(200):  # max steps per episode
            s_tensor = torch.tensor(np.array(s), dtype=torch.float32)
            mu, sigma, value = local_net(s_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            a = dist.sample().detach().numpy()[0]
            a = np.clip(a, -2, 2)

            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_reward += r

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)  # scale reward

            if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                # bootstrap value
                if done:
                    v_s_ = 0
                else:
                    s_tensor_ = torch.tensor(np.array(s_), dtype=torch.float32)
                    _, _, v_s_ = local_net(s_tensor_)
                    v_s_ = v_s_.detach().item()

                # discounted returns
                buffer_v_target = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()

                bs = torch.tensor(np.array(buffer_s), dtype=torch.float32)
                ba = torch.tensor(np.array(buffer_a), dtype=torch.float32)
                bv = torch.tensor(buffer_v_target, dtype=torch.float32)

                mu, sigma, values = local_net(bs)
                dist = torch.distributions.Normal(mu, sigma)
                log_prob = dist.log_prob(ba)
                entropy = dist.entropy()

                advantage = bv - values.squeeze(1)

                policy_loss = -(log_prob * advantage.detach()).mean()
                value_loss = F.mse_loss(values.squeeze(1), bv)
                entropy_loss = -ENTROPY_BETA * entropy.mean()

                total_loss = policy_loss + value_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()

                for lp, gp in zip(local_net.parameters(), global_net.parameters()):
                    gp._grad = lp.grad
                optimizer.step()

                local_net.load_state_dict(global_net.state_dict())

                buffer_s, buffer_a, buffer_r = [], [], []

            s = s_
            total_step += 1
            if done:
                with global_ep.get_lock():
                    global_ep.value += 1
                res_queue.put(ep_reward)
                break

    res_queue.put(None)


# ---------------- Main Train ---------------- #
def train():
    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    global_net = Net(s_dim, a_dim)
    global_net.share_memory()
    optimizer = torch.optim.Adam(global_net.parameters(), lr=LR)

    global_ep = mp.Value("i", 0)
    res_queue = mp.Queue()

    workers = []
    for i in range(NUM_WORKERS):  # 👈 تعداد worker مشخص‌شده
        p = mp.Process(
            target=worker, args=(global_net, optimizer, global_ep, res_queue, i)
        )
        p.start()
        workers.append(p)

    results = []
    while True:
        r = res_queue.get()
        if r is None:
            break
        results.append(r)

        ep_num = len(results)

        if ep_num % PRINT_EVERY == 0:
            avg_reward = np.mean(results[-PRINT_EVERY:])
            print(
                f"Episode: {ep_num}, "
                f"Last Reward: {r:.2f}, "
                f"Avg Reward (last {PRINT_EVERY}): {avg_reward:.2f}"
            )

    for p in workers:
        p.join()

    return results


if __name__ == "__main__":
    train()