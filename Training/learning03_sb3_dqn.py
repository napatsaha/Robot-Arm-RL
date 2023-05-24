# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:36:18 2023

@author: napat
"""

import os
import gym
from gym.spaces import MultiDiscrete, Box
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN

# a = MultiDiscrete(np.full((7,), 3))
# o = Box(low=-1.0, high=1.0, shape=(9,))

class UnityGym(gym.Env):
    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.behaviour_name = list(env.behavior_specs.keys())[0]
        self.spec = env.behavior_specs.get(self.behaviour_name)
        self.n_actions = self.spec.action_spec.discrete_branches[0]
        self.n_branches = self.spec.action_spec.discrete_size
        self.n_observations = self.spec.observation_specs[0].shape[0]
        self.action_space = MultiDiscrete(np.full((self.n_branches,), self.n_actions))
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.n_observations,))
        
        
    def reset(self):
        self.env.reset()
        state, _ = self._get_state()
        return state
        
    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = np.expand_dims(action, axis=0) # Add extra axis
        action = ActionTuple(discrete=action)
        self.env.set_actions(self.behaviour_name, action)
        self.env.step()
        done = self._is_done()
        state, reward = self._get_state()
        return state, reward, done, {}
        
    def _is_done(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)
        return len(terminal_steps.reward) > 0
    
    def _get_state(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behaviour_name)
        if self._is_done():
            state = terminal_steps.obs[0]
            reward = terminal_steps.reward
        else:
            state = decision_steps.obs[0]
            reward = decision_steps.reward
            
        state = state.reshape(-1)
        reward = reward.item()
        return state, reward
    
    def random_actions(self):
        action_spec = self.spec.action_spec
        return action_spec.random_action(1).discrete.reshape(-1)
        
    def close(self):
        self.env.close()

def evaluate(env, model, n_episode=20):
    rewards = []
    steps = []
    for i in range(n_episode):
        state = env.reset()
        done = False
        eps_reward = 0
        n_steps = 0
        while not done:
            action, _ = model.predict(state)
            nextstate, reward, done, _ = env.step(action)
            eps_reward += reward
            n_steps += 1
            
        rewards.append(eps_reward)
        steps.append(n_steps)
        print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Length: {n_steps}')
    rewards = np.array(rewards)
    steps = np.array(steps)
    return rewards, steps

if __name__ == "__main__":
    try:
        trial = 1
        unityenv = UnityEnvironment()
        env = UnityGym(unityenv)
        print(check_env(env))
        model = PPO('MlpPolicy', env, verbose=2, tensorboard_log="results")
        
        model.learn(total_timesteps=20000, tb_log_name='sb3'+str(trial).zfill(2))
    
        model.save(os.path.join('Training','Model','model_sb3_'+str(trial).zfill(2)))
    
        history, length = evaluate(env, model)
        succeed = (length < max(length)).mean()
        print(f'Average reward: {history.mean():.3f}')
        print(f'Proportion of episodes succeeded: {succeed:.2%}')
        
    finally:
        unityenv.close()
