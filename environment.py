import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudResourceEnv(gym.Env):
    def __init__(self):
        super(CloudResourceEnv, self).__init__()
        
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        self.state = [0.5, 0.5, 0.5, 0.5] 
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        self.state = [0.5, 0.5, 0.5, 0.5]
        self.steps = 0
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        cpu_scale, mem_scale = action
        dynamic_load = np.random.uniform(0.3, 0.8)
        cpu_usage = dynamic_load * cpu_scale
        mem_usage = dynamic_load * mem_scale
        reward = -np.abs(cpu_usage - cpu_scale) - np.abs(mem_usage - mem_scale)
        self.state = [cpu_usage, mem_usage, dynamic_load, cpu_scale]
        done = self.steps >= self.max_steps
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.steps}, State: {self.state}")
