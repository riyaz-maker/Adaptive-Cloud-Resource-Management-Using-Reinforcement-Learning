from stable_baselines3 import PPO
from environment import CloudResourceEnv

env = CloudResourceEnv()

model = PPO.load("models/ppo_model")

obs, _ = env.reset()
total_reward = 0
for step in range(200):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if done:
        break

print(f"Total reward after evaluation: {total_reward}")
