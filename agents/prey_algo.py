import numpy as np

class Prey:
    def __init__(self):
        print("Prey initialized (MPE Version - 5 Actions).")

    def get_action(self, obs):
        rel_pred1 = obs[8:10]
        rel_pred2 = obs[10:12]
        rel_pred3 = obs[12:14]
        
        escape_vec = - (rel_pred1 + rel_pred2 + rel_pred3)
        norm = np.linalg.norm(escape_vec)
        if norm > 0:
            direction = escape_vec / norm
        else:
            direction = np.random.uniform(-1.0, 1.0, size=2)
            
        action = np.zeros(5, dtype=np.float32)
        power = 1.0
        if direction[0] > 0:
            action[1] = direction[0] * power
        else:
            action[2] = -direction[0] * power
            
        if direction[1] > 0:
            action[3] = direction[1] * power
        else:
            action[4] = -direction[1] * power
        return action