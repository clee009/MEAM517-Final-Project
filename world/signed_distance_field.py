import numpy as np
from world import Obstacles
import matplotlib.pyplot as plt


class SignedDistanceField(Obstacles):
    def __init__(self, file: str, epsilon = 5e-6):
        super().__init__(file)
        self.n = len(self.boxes)
        self.eps = epsilon
    

    def __calc_sdf_single(self, state, idx):
        x, y = state[:2]
        x_min, y_min, x_max, y_max = self.boxes[idx]

        if x < x_min:
            if y < y_min:
                return np.hypot(x_min - x, y_min - y)  # Bottom-left corner
            elif y > y_max:
                return np.hypot(x_min - x, y - y_max)  # Top-left corner
            else:
                return x_min - x  # Left edge
            
        elif x > x_max:
            if y < y_min:
                return np.hypot(x - x_max, y_min - y)  # Bottom-right corner
            elif y > y_max:
                return np.hypot(x - x_max, y - y_max) # Top-right corner
            else:
                return x - x_max  # Right edge
            
        else:
            if y < y_min:
                return y_min - y  # Bottom edge
            elif y > y_max:
                return y - y_max  # Top edge
            else:
                return max(x - x_max, x_min - x, y - y_max, y_min - y)

    

    def calc_sdf(self, x):
        min_sdf = -self.__calc_sdf_single(x, 0)

        for i in range(1, self.n):
            min_sdf = min(self.__calc_sdf_single(x, i), min_sdf)
        
        return min_sdf
    

    def calc_ellips(self, state, box, lambda_param):
        x, y = state[0, 0], state[0, 1]

        xmin, ymin, xmax, ymax = box

        # Define numerical values for the parameters
        c_x = (xmin + xmax) / 2   # Numerical value for the center x
        c_y = (ymin + ymax) / 2  # Numerical value for the center y
        r_x = (xmax - xmin) / 2  # Numerical value for the size in x
        r_y = (ymax - ymin) / 2  # Numerical value for the size in y

        # Define the cost function directly using numerical values
        penalty = 1 / (((x - c_x)**2 / r_x + (y - c_y)**2 / r_y)**lambda_param + 1)
    

    def calc_grad(self, x: np.ndarray):
        grad = np.zeros((2,))
        for i in range(2):
            xpos = x[:2].copy()
            xneg = x[:2].copy()
            xpos[i] += self.eps
            xneg[i] -= self.eps
            grad[i] = self.calc_sdf(xpos) - self.calc_sdf(xneg)

        grad /= 2* self.eps
        return grad
    
    
    def plot_barrier(self):
        x_min, y_min, x_max, y_max = self.boxes[0]

        # Generate a grid of points
        x_vals = np.linspace(x_min, x_max, int(100 * (x_max - x_min)))
        y_vals = np.linspace(y_max, y_min, int(100 * (y_max - y_min)))
        X, Y = np.meshgrid(x_vals, y_vals)

        # Calculate the signed distance for each point in the grid
        sdf_values = np.vectorize(lambda x, y: self.barrier_func([x, y]))(X, Y)

        # Plot the signed distance field
        plt.imshow(sdf_values, cmap='jet')
        plt.colorbar(label='Signed Distance')
        plt.title('Signed Distance Field to Rectangle')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


    def barrier_func(self, x):
        sdf = self.calc_sdf(x)
        return np.exp(-sdf)