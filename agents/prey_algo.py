import numpy as np
import math

class Prey:
    def __init__(self):
        print("Prey algorithm initialized. (Advanced Radar 2.0)")

    def get_action(self, obs=None):
        if obs is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        prey_pos = obs[-2:]
        
        predators = []
        for i in range(0, len(obs) - 2, 2):
            if obs[i] != -1.0:
                predators.append(obs[i:i+2])
                
        if not predators:
            return np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

        num_directions = 32
        best_action = np.array([0.0, 0.0])
        max_score = -float('inf')
        
        look_ahead = 0.15 
        
        for angle in np.linspace(0, 2 * math.pi, num_directions, endpoint=False):
            dir_vec = np.array([math.cos(angle), math.sin(angle)])
            future_pos = prey_pos + dir_vec * look_ahead
            
            margin = 0.00
            if (future_pos[0] < margin or future_pos[0] > 1.0 - margin or 
                future_pos[1] < margin or future_pos[1] > 1.0 - margin):
                continue 
            
            # TIÊU CHÍ 1: Tránh xa cá mập (Trọng số cao)
            min_pred_dist = min([np.linalg.norm(future_pos - p) for p in predators])

            dist_to_walls = min(future_pos[0], 1.0 - future_pos[0], future_pos[1], 1.0 - future_pos[1])
            
            # TỔNG ĐIỂM: Khuyến khích né cá mập VÀ vòng ra giữa sân
            score = min_pred_dist * 2.0 + dist_to_walls * 1.0
            
            if score > max_score:
                max_score = score
                best_action = dir_vec

        if max_score == -float('inf'):
            center = np.array([0.5, 0.5])
            escape_vec = center - prey_pos
            norm = np.linalg.norm(escape_vec)
            if norm > 0:
                return (escape_vec / norm).astype(np.float32)
            else:
                return np.array([0.0, 0.0], dtype=np.float32)

        return best_action.astype(np.float32)