# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:10:58 2023

@author: napat
"""

import gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
rewards = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    rewards += reward
    #vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

# rewards = np.array(rewards)
print(rewards)

env.close()