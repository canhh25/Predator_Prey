import functools
import numpy as np
from pettingzoo import ParrallelEnv
from gymnasium import Discrete, Box

class PreyAlgoEnv(ParrallelEnv):
    metadata = {'render.modes': ['text'], "name":"gridworld_v0"}

    def __init__(self, grid_size = 8, num_predator =3):
        self.grid_size = grid_size
        self.num_predator = num_predator
        self.num_agents = [f"predator_{i}" for i in range(num_predator)]
        self.agents = self.num_agents[:]
        self.predator_pos = {}
        self.prey_pos = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)  # 0: stay, 1: up, 2: down, 3: left, 4: right
    
    def reset(self, seed=None, options=None):
        self.agents = self.num_agents[:]
        for i, agent in enumerate(self.agents):
            self.predator_pos[agent] = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))   
        self.prey_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        observations = {agent: np.array(self.predator_pos[agent]) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: -0.1 for agent in self.agents}
        if any(terminations.values()):
            self.agents=[]
        observations = {agents: self._get_obs(agent) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos
    
    def _get_obs(self, agent):
        obs=[]
        for agent in self.num_agents:
            obs.extend(self.predator_pos.get([agent],[-1,-1]))
        obs.extend(self.prey_pos)
        return np.array(obs, dtype=np.int32)