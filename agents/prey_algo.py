import numpy as np

class Prey:
    def __init__(self):
        print("Prey here hehee catch me if u can.")

    def get_action(self, obs):
        my_pos = obs[-4:-2]
        predators = [obs[8:10], obs[10:12], obs[12:14]]
        force_pred = np.zeros(2)
        for p_pos in predators:
            dist = np.linalg.norm(my_pos-p_pos)
            if dist < 0.6:
                force_pred += (p_pos - my_pos) / (dist**2 + 0.001)
        force_wall = np.zeros(2)
        margin = 0.2
        if my_pos[0] < margin: force_wall[0] += (margin - my_pos[0]) * 5
        if my_pos[0] > 1.0 - margin: force_wall[0] -= (my_pos[0] - (1.0 - margin)) * 5
        if my_pos[1] < margin: force_wall[1] += (margin - my_pos[1]) * 5
        if my_pos[1] > 1.0 - margin: force_wall[1] -= (my_pos[1] - (1.0 - margin)) * 5    
        total_force = force_pred + force_wall
        norm = np.linalg.norm(total_force)
        if norm > 0:
            direction = total_force / norm
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