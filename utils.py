# utils.py
import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window)/window, mode="valid")

def plot_results(a2c_rewards, a3c_rewards, times, window=20, savepath=None):
    ma_a2c = moving_average(a2c_rewards, window)
    ma_a3c = moving_average(a3c_rewards, window)

    plt.figure(figsize=(10, 5))
    plt.plot(ma_a2c, label="A2C")
    plt.plot(ma_a3c, label="A3C")
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward")
    plt.title(f"A2C vs A3C (window={window})")
    plt.legend()
    if savepath:
        plt.savefig(savepath + "_rewards.png")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.bar(["A2C", "A3C"], times, color=["blue", "orange"])
    plt.ylabel("Time (s)")
    plt.title("Execution Time Comparison")
    if savepath:
        plt.savefig(savepath + "_times.png")
    plt.show()