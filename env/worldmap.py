import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import pygame
import math
class WorldMap(ParallelEnv):
    metadata = {'render_modes': ['human'], "name":"WorldMap_v0"}

    def __init__(self, num_predator = 3):
        self.max_size = 1.0
        self.num_predator = num_predator
        self.possible_agents = [f"predator_{i}" for i in range(num_predator)]
        self.agents = self.possible_agents[:]
        self.predator_pos = {}
        self.prey_pos = None    
        self.predator_radius = 0.05
        self.prey_radius = 0.04
        self.action_spacess = {}
        self.observation_spaces = {}
        obs_dim=2*self.num_predator+2
        for agent in self.possible_agents:
            self.observation_spaces[agent] = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self.observation_spaces[agent] = Box(low=0.0, high=self.max_size, shape=(obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_length = 2 * self.num_predator + 2
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.predator_pos = {}
        for i, agent in enumerate(self.agents):
            self.predator_pos[agent] = np.random.uniform(0.0, self.max_size, size=(2,)).astype(np.float32)
        self.prey_pos = np.random.uniform(0.0, self.max_size, size=(2,)).astype(np.float32)
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        dt=0.005
        prey_action = actions.pop("prey", np.array([0.0, 0.0], dtype=np.float32))
        for agent, action in actions.items(): 
            self.predator_pos[agent] += action * dt
            self.predator_pos[agent] = np.clip(self.predator_pos[agent], 0.0, self.max_size)
        self.prey_pos += prey_action * dt
        self.prey_pos = np.clip(self.prey_pos, 0.0, self.max_size)
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: -0.1 for agent in self.agents}
        is_caught = False
        for agent in self.agents:
            dist = math.hypot(self.predator_pos[agent][0] - self.prey_pos[0],  self.predator_pos[agent][1] - self.prey_pos[1])
            if dist < self.predator_radius + self.prey_radius:
                is_caught = True
                break
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        if is_caught:
            for agent in self.agents:
                rewards[agent] += 10.0
                terminations[agent] = True
            self.agents = []
        return observations, rewards, terminations, truncations, infos
    
    def _get_obs(self, agent):
        obs = []
        for pred_name in self.possible_agents:
            if pred_name in self.predator_pos:
                obs.extend(self.predator_pos[pred_name])
            else:
                obs.extend([-1.0, -1.0]) # Nếu chết thì gửi tọa độ ảo
                
        if self.prey_pos is not None:
            obs.extend(self.prey_pos)
        else:
            obs.extend([-1.0, -1.0])
            
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        window_size = 512
        scale = window_size / self.max_size 

        if not hasattr(self, 'window'):
            pygame.init()
            pygame.display.set_caption("Predator vs Prey - Continuous Space")
            self.window = pygame.display.set_mode((window_size, window_size))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', int(self.predator_radius * scale), bold=True)

        self.window.fill((255, 255, 255))

        if self.prey_pos is not None:
            px, py = int(self.prey_pos[0] * scale), int(self.prey_pos[1] * scale)
            r_pixel = int(self.prey_radius * scale)
            pygame.draw.circle(self.window, (0, 150, 255), (px, py), r_pixel)
            
            prey_text = self.font.render("P", True, (255, 255, 255))
            self.window.blit(prey_text, prey_text.get_rect(center=(px, py)))

        for agent, pos in self.predator_pos.items():
            px, py = int(pos[0] * scale), int(pos[1] * scale)
            r_pixel = int(self.predator_radius * scale)
            idx = int(agent.split('_')[1])
            
            pygame.draw.circle(self.window, (255, 50, 50), (px, py), r_pixel)
            
            text_surface = self.font.render(str(idx), True, (255, 255, 255))
            self.window.blit(text_surface, text_surface.get_rect(center=(px, py)))

        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()