import numpy as np
class Prey:
    def __init__(self):
        print("Prey algorithm initialized.")
    def get_action(self,obs=None):
        prey_x = obs[-2]
        prey_y = obs[-1]
        min_dist = float('inf')
        closest_x = -1
        closest_y = -1
        for i in range(0,len(obs)-2,2):
            predator_x = obs[i]
            predator_y = obs[i+1]
            if predator_x == -1:
                continue
            dist = self.distance(prey_x,prey_y, predator_x, predator_y)
            if dist < min_dist:
                min_dist = dist
                closest_x = predator_x
                closest_y = predator_y
        if closest_x == -1:
            return np.random.choice([0, 1, 2, 3, 4])
        elif closest_x < prey_x:
            return 2
        elif closest_x > prey_x:
            return 1
        elif closest_y < prey_y:
            return 4
        else:
            return 3
    def distance(self, pos1_x, pos1_y, pos2_x, pos2_y):
        return abs(pos1_x - pos2_x) + abs(pos1_y - pos2_y)