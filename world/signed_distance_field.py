import numpy as np
from world import Obstacles
import matplotlib.pyplot as plt


class SignedDistanceField(Obstacles):
    def __init__(self, file: str):
        super().__init__(file)
        self.n = len(self.boxes)
    

    def signed_distance_to_rectangle(self, state, idx):
        x_point, y_point = state[:2]
        x_min, y_min, x_max, y_max = self.boxes[idx]

        if x_point < x_min:
            if y_point < y_min:
                return ((x_min - x_point)**2 + (y_min - y_point)**2)**0.5  # Bottom-left corner
            elif y_point > y_max:
                return ((x_min - x_point)**2 + (y_point - y_max)**2)**0.5  # Top-left corner
            else:
                return x_min - x_point  # Left edge
        elif x_point > x_max:
            if y_point < y_min:
                return ((x_point - x_max)**2 + (y_min - y_point)**2)**0.5  # Bottom-right corner
            elif y_point > y_max:
                return ((x_point - x_max)**2 + (y_point - y_max)**2)**0.5  # Top-right corner
            else:
                return x_point - x_max  # Right edge
        else:
            if y_point < y_min:
                return y_min - y_point  # Bottom edge
            elif y_point > y_max:
                return y_point - y_max  # Top edge
            else:
                x, y, = x_point, y_point
                return -min(x_max - x, y_max - y, x - x_min, y - y_min)

    

    def calc_sdf(self, state):
        min_sdf = np.inf
        for i in range(self.n):
            sdf = self.signed_distance_to_rectangle(state, i)
            min_sdf = min(min_sdf, sdf if i else -sdf)

        return min_sdf
    
    def plot_sdf(self):
        x_min, y_min, x_max, y_max = self.boxes[0]

        # Generate a grid of points
        x_vals = np.linspace(x_min, x_max, int(100 * (x_max - x_min)))
        y_vals = np.linspace(y_max, y_min, int(100 * (y_max - y_min)))
        X, Y = np.meshgrid(x_vals, y_vals)

        # Calculate the signed distance for each point in the grid
        sdf_values = np.vectorize(lambda x, y: self.calc_sdf([x, y]))(X, Y)

        # Plot the signed distance field
        plt.imshow(sdf_values, cmap='jet')
        plt.colorbar(label='Signed Distance')
        plt.title('Signed Distance Field to Rectangle')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


    def barrier_gradient(self, x):
        sdf, gradient_sdf = self.signed_distance_field_and_gradient(x)
        gradient_barrier = -gradient_sdf / self.epsilon * np.exp(-sdf / self.epsilon)
        return gradient_barrier
    
