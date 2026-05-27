import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import pygame
class GridWorldEnv(ParallelEnv):
    metadata = {'render_modes': ['text'], "name":"gridworld_v0"}

    def __init__(self, grid_size = 8, num_predator = 5):
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
        window_size = 512
        cell_size = window_size // self.grid_size

        if not hasattr(self, 'window'):
            pygame.init()
            pygame.display.set_caption("Predator vs Prey - Multi-Agent RL")
            self.window = pygame.display.set_mode((window_size, window_size))
            self.clock = pygame.time.Clock()
            
            # --- MỚI: Khởi tạo bộ Font chữ để vẽ số ---
            pygame.font.init()
            # Kích thước chữ bằng một nửa ô vuông cho vừa vặn
            self.font = pygame.font.SysFont('Arial', cell_size // 2, bold=True)

        self.window.fill((255, 255, 255))

        # Vẽ lưới
        for x in range(0, window_size, cell_size):
            pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, window_size))
        for y in range(0, window_size, cell_size):
            pygame.draw.line(self.window, (200, 200, 200), (0, y), (window_size, y))

        # --- MỚI: Bảng màu cho các con cá mập ---
        # (Đỏ, Tím, Cam, Xanh lá, Hồng...) Bạn có thể thêm bao nhiêu màu tùy thích
        predator_colors = [(255, 50, 50)]

        # Vẽ cá mập
        for agent, (x, y) in self.predator_pos.items():
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            
            # Lấy số ID của cá mập (VD: "predator_1" -> lấy số 1)
            idx = int(agent.split('_')[1])
            
            # Chọn màu dựa theo ID (dùng % để nếu số lượng cá mập nhiều hơn số màu thì nó tự lặp lại)
            color = predator_colors[idx % len(predator_colors)]
            
            # 1. Vẽ hình vuông với màu riêng
            pygame.draw.rect(self.window, color, rect)
            
            # 2. Vẽ số ID (màu trắng) đè lên trên hình vuông
            text_surface = self.font.render(str(idx), True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=rect.center) # Căn giữa ô vuông
            self.window.blit(text_surface, text_rect)

        # Vẽ con mồi
        if self.prey_pos:
            px, py = self.prey_pos
            center = (py * cell_size + cell_size // 2, px * cell_size + cell_size // 2)
            radius = cell_size // 2 - 4
            
            # Vẽ hình tròn màu Xanh dương cho con mồi
            pygame.draw.circle(self.window, (0, 150, 255), center, radius)
            
            # Vẽ thêm chữ "P" lên người con mồi cho ngầu
            prey_text = self.font.render("P", True, (255, 255, 255))
            prey_rect = prey_text.get_rect(center=center)
            self.window.blit(prey_text, prey_rect)

        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    print()