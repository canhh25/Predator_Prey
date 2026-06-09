import numpy as np

class Predator:
    def get_action(self, obs):
        prey_pos = obs[-2:] 
        norm = np.linalg.norm(prey_pos)
        if norm > 0:
            direction = prey_pos / norm
        else:
            direction = np.array([0.0, 0.0])
            
        action = np.zeros(5, dtype=np.float32)
        power = 0.8
        
        if direction[0] > 0:
            action[1] = direction[0] * power  
        else:
            action[2] = -direction[0] * power
            
        if direction[1] > 0:
            action[3] = direction[1] * power  
        else:
            action[4] = -direction[1] * power
        return action