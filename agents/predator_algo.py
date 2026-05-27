import numpy as np
class Predator:
    def __init__(self):
        print("Predator algorithm initialized.")
    def get_action(self,obs=None):
        return np.random.choice([0, 1, 2, 3, 4])