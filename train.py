
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# 1. Create the environment
env = gym.make(
    "CarRacing-v3",
    continuous=False
)

env = Monitor(env, filename="training_log.csv")

# 2. Initialize DQN
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=0.0001,#how fast the weights are updated
    buffer_size=10000,#how many previous can it store
    learning_starts=1000,# when does it starts updating the nn weights
    batch_size=32,
    gamma=0.99,#discount factors for calculating q values
    train_freq=9,
    gradient_steps=1,
    target_update_interval=5000,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    verbose=1,
    tensorboard_log="./dqn_tensorboard/"
)
# 3. Train
model.learn(
    total_timesteps=50000,
    tb_log_name="CarRacing_DQN",
    log_interval=10,
    progress_bar=True
)

# 4. Save the trained model
model.save("car_racing_dqn")

env.close()
