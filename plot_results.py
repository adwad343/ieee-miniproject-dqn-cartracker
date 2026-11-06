import pandas as pd
import matplotlib.pyplot as plt


# Read training and test data
training_df = pd.read_csv("training_log.csv.monitor.csv", skiprows=1)
test_df = pd.read_csv("test_log.csv.monitor.csv", skiprows=1)

# Plot rewards for both phases
plt.figure(figsize=(12, 5))

plt.plot(range(1, len(training_df) + 1), training_df["r"], label="Training Reward")
plt.plot(range(1, len(test_df) + 1), test_df["r"], label="Testing Reward")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN CarRacing - Training and Testing Rewards per Episode")

plt.legend()
plt.grid()
plt.savefig("episode_rewards.png")
plt.show()


# for avg rewards over 100 episode 
test_episodes = range(1, len(test_df) + 1)
test_rewards = test_df["r"]
moving_avg = test_rewards.rolling(window=100, min_periods=1).mean()

plt.figure(figsize=(12, 5))

plt.plot(test_episodes, moving_avg, label="100 Episode Moving Average")

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("DQN CarRacing - Test Moving Average Reward")

plt.legend()
plt.grid()
plt.savefig("moving_average.png")
plt.show()


# for episode length
lengths = test_df["l"]

plt.figure(figsize=(12, 5))

plt.plot(test_episodes, lengths, label="Episode Length")

plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.title("DQN CarRacing - Test Episode Length")

plt.legend()
plt.grid()
plt.savefig("episode_length.png")
plt.show()