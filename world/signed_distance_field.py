import numpy as np
from world import Obstacles
import matplotlib.pyplot as plt
import torch


class SignedDistanceField(Obstacles):
    def __init__(self, file: str, gamma = 3.6):
        super().__init__(file)
        self.n = len(self.boxes)
        self.gamma = gamma
    

    def calc_sdf_single(self, state: torch.Tensor, idx):
        x, y = state[:2]
        x_min, y_min, x_max, y_max = self.boxes[idx]

        if x < x_min:
            if y < y_min:
                return torch.hypot(x_min - x, y_min - y)  # Bottom-left corner
            elif y > y_max:
                return torch.hypot(x_min - x, y - y_max)  # Top-left corner
            else:
                return x_min - x  # Left edge
            
        elif x > x_max:
            if y < y_min:
                return torch.hypot(x - x_max, y_min - y)  # Bottom-right corner
            elif y > y_max:
                return torch.hypot(x - x_max, y - y_max) # Top-right corner
            else:
                return x - x_max  # Right edge
            
        else:
            if y < y_min:
                return y_min - y  # Bottom edge
            elif y > y_max:
                return y - y_max  # Top edge
            else:
                dx = torch.max(x - x_max, x_min - x)
                dy = torch.max(y - y_max, y_min - y)
                return torch.max(dx, dy)

    

    def calc_sdf(self, x: torch.Tensor):
        min_sdf = -self.calc_sdf_single(x, 0)
        for i in range(1, self.n):
            min_sdf = torch.min(self.calc_sdf_single(x, i), min_sdf)
        
        return min_sdf
    
    
    def plot_barrier(self):
        x_min, y_min, x_max, y_max = self.boxes[0]

        # Generate a grid of points
        x_vals = np.linspace(x_min, x_max, int(10 * (x_max - x_min)))
        y_vals = np.linspace(y_max, y_min, int(10 * (y_max - y_min)))
        X, Y = np.meshgrid(x_vals, y_vals)

        # Calculate the signed distance for each point in the grid
        sdf_values = np.vectorize(lambda x, y: self.barrier_func(torch.Tensor([x, y])).item())(X, Y)

        # Plot the signed distance field
        plt.imshow(sdf_values, cmap='jet')
        plt.colorbar(label='Signed Distance')
        plt.title('Signed Distance Field to Rectangle')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


    def barrier_func(self, x):
        sdf = self.calc_sdf(x)
        return torch.exp(-self.gamma * sdf)