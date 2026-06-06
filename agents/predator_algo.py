import numpy as np

class Predator:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.idx = int(agent_name.split('_')[1])

    def get_action(self, obs):
        my_pos = obs[self.idx * 2 : self.idx * 2 + 2]
        prey_pos = obs[-2:]
        direction = prey_pos - my_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            action = direction / norm * 0.6
        else:
            action = np.array([0.0, 0.0])
        return action.astype(np.float32)