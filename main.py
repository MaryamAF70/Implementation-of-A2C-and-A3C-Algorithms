import time
import numpy as np
from a2c import train_a2c
from a3c import train_a3c
from utils import plot_results

# -------------------------------
# هایپرپارامترها
# -------------------------------
class Args:
    env_name = "Pendulum-v1"

    # عمومی
    max_episodes = 80000       # تعداد کل اپیزودها
    gamma = 0.99               # ضریب تنزیل
    lr = 3e-4                  # نرخ یادگیری
    hidden_dim = 128           # اندازه لایه‌های مخفی
    log_interval = 1000        # فاصله نمایش لاگ‌ها
    window = 20                # پنجره میانگین متحرک

    # مخصوص A2C
    update_interval = 5

    # مخصوص A3C
    num_processes = 4          # تعداد پروسس‌ها


args = Args()

# -------------------------------
# اجرای آزمایش‌ها
# -------------------------------
def run_experiments():
    # -------- A3C --------
    print("Running A3C ...")
    a3c_rewards, a3c_time = train_a3c(args)
    a3c_mean = np.mean(a3c_rewards[-min(100, len(a3c_rewards)):]) if len(a3c_rewards) > 0 else float("nan")
    print(f"A3C done. Episodes: {len(a3c_rewards)}. Time: {a3c_time:.2f}s")
    print(f"A3C mean reward (last 100): {a3c_mean:.3f}")

    # -------- A2C --------
    print("Running A2C ...")
    a2c_rewards, a2c_time = train_a2c(args)
    a2c_mean = np.mean(a2c_rewards[-min(100, len(a2c_rewards)):])
    print(f"A2C done. Episodes: {len(a2c_rewards)}. Time: {a2c_time:.2f}s")
    print(f"A2C mean reward (last 100): {a2c_mean:.3f}")

    # -------- Summary --------
    print("\nSummary:")
    print("Algorithm | Episodes | MeanReward(100) | Time(s)")
    print(f"A3C       | {len(a3c_rewards):7d} | {a3c_mean:15.3f} | {a3c_time:7.2f}")
    print(f"A2C       | {len(a2c_rewards):7d} | {a2c_mean:15.3f} | {a2c_time:7.2f}")

    # -------- Plots --------
    plot_results(a2c_rewards, a3c_rewards,
                 times=[a2c_time, a3c_time],
                 window=args.window,
                 savepath="results_pendulum")


if __name__ == "__main__":
    run_experiments()