
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# 1. Create the environment with human rendering.
env = gym.make(
    "CarRacing-v3",
    continuous=False,
)
env = Monitor(env, filename="test_log.csv")

# 2. Load the trained model.
model = DQN.load("car_racing_dqn")

# 3. Run the trained agent.
obs, info = env.reset()

episodes = 100

for episode in range(episodes):

    obs, info = env.reset()
    terminated = False
    truncated = False

    total_reward = 0

    while not (terminated or truncated):

        # Predict the action using the trained neural network.
        action, _states = model.predict(
            obs,
            deterministic=True
        )

        # Execute the selected action.
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Small delay to make viewing easier.
        #time.sleep(0.01)

    print(
        f"Episode {episode + 1}: "
        f"Total Reward = {total_reward:.2f}"
    )

env.close()
