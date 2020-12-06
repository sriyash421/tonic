'''Environment wrappers.'''

import gym
import numpy as np


class ActionRescaler(gym.ActionWrapper):
    '''Rescales actions from [-1, 1]^n to the true action space.
    The baseline agents return actions in [-1, 1]^n.'''

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        high = np.ones(env.action_space.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-high, high=high)
        true_low = env.action_space.low
        true_high = env.action_space.high
        self.bias = (true_high + true_low) / 2
        self.scale = (true_high - true_low) / 2

    def action(self, action):
        return self.bias + self.scale * np.clip(action, -1, 1)


class TimeFeature(gym.Wrapper):
    '''Adds a notion of time in the observations.
    It can be used in terminal timeout settings to get Markovian MDPs.
    '''

    def __init__(self, env, max_steps, low=-1, high=1):
        super().__init__(env)
        dtype = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, low).astype(dtype),
            high=np.append(self.observation_space.high, high).astype(dtype))
        self.max_episode_steps = max_steps
        self.steps = 0
        self.low = low
        self.high = high

    def reset(self, **kwargs):
        self.steps = 0
        observation = self.env.reset(**kwargs)
        observation = np.append(observation, self.low)
        return observation

    def step(self, action):
        assert self.steps < self.max_episode_steps
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        prop = self.steps / self.max_episode_steps
        v = self.low + (self.high - self.low) * prop
        observation = np.append(observation, v)
        return observation, reward, done, info

class Recurrent(gym.Wrapper):
    
    def __init__(self, env, max_steps):
        self.max_history = max_steps
        super().__init__(env)
        dtype = self.observation_space.dtype
        self._shape = (self.max_history,)+self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=self.observation_space.low, high=self.observation_space.high)
        self.steps = 0
        self.low = self.observation_space.low
        self.high = self.observation_space.high

    def reset(self, **kwargs):
        self.steps = 0
        observation = self.env.reset(**kwargs)
        self.current_observation = np.zeros(self._shape)
        self.current_observation[-1,:] = observation
        return self.current_observation

    def step(self, action):
        assert self.steps < self.max_episode_steps
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        self.current_observation = np.append(self.current_observation[1:,:], observation[None,:], axis=0)
        return self.current_observation, reward, done, info
