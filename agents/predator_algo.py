import numpy as np

class Predator:
    def __init__(self, agent_id):
        self.agent_id = agent_id
    def get_action(self, obs):
        direction = np.random.uniform(-1.0, 1.0, size=2)
        action = np.zeros(5, dtype=np.float32)
        power = 1.0
        if direction[0] > 0:
            action[2] = direction[0] * power 
        else:
            action[1] = -direction[0] * power  
        if direction[1] > 0:
            action[4] = direction[1] * power 
        else:
            action[3] = -direction[1] * power
        return action