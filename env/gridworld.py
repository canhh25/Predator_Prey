import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
class GridWorldEnv(ParallelEnv):
    metadata = {'render_modes': ['text'], "name":"gridworld_v0"}

    def __init__(self, grid_size = 8, num_predator = 10):
        self.grid_size = grid_size
        self.num_predator = num_predator
        self.possible_agents = [f"predator_{i}" for i in range(num_predator)]
        self.agents = self.possible_agents[:]
        self.predator_pos = {}
        self.prey_pos = None    

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_length = 2 * self.num_predator + 2
        return Box(low=0, high=self.grid_size-1, shape=(obs_length,), dtype=np.int32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)  # 0: stay, 1: up, 2: down, 3: left, 4: right
    
    def reset(self, seed=None, options=None):
        self.predator_pos = {}
        self.agents = self.possible_agents[:]
        for i, agent in enumerate(self.agents):
            self.predator_pos[agent] = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))   
        self.prey_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos
    
    def move(self, pos, action):
        move_map = {
            0: (0, 0),   # stay 
            1: (-1, 0),  # up
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 1)    # right
        }
        dx, dy = move_map[action]
        x, y = pos
        new_x, new_y = max(0,min(self.grid_size-1,x+dx)), max(0,min(self.grid_size-1,y+dy))
        return (new_x,new_y)

    def step(self, actions):
        prey_action = actions.pop("prey", np.random.choice([0, 1, 2, 3, 4]))
        for agent, action in actions.items(): 
            self.predator_pos[agent] = self.move(self.predator_pos[agent], action)
        self.prey_pos = self.move(self.prey_pos, prey_action)
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: -0.1 for agent in self.agents}
        is_caught = False
        for agent in self.agents:
            if self.predator_pos[agent] == self.prey_pos:
                rewards[agent] += 10.0
                is_caught = True
                break
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        if is_caught:
            for agent in self.agents:
                terminations[agent] = True
            self.agents = []
        return observations, rewards, terminations, truncations, infos
    
    def _get_obs(self, agent):
        obs=[]
        for pred_name in self.possible_agents:
            obs.extend(self.predator_pos.get(pred_name, [-1, -1]))
        obs.extend(self.prey_pos)
        return np.array(obs, dtype=np.int32)
    
    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        px, py = self.prey_pos
        grid[px][py] = "P"
    
        for agent, (x, y) in self.predator_pos.items():
            grid[x][y] = "X"

        for row in grid:
            print(" ".join(row))

    print()