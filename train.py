from stable_baselines3 import PPO
from environment import CloudResourceEnv

env = CloudResourceEnv()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

model.save("models/ppo_model")
print("Model trained and saved.")
