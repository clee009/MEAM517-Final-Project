import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

import configs
import pydrake.symbolic as sym


class Obstacles:
    def __init__(self, file: str):
        self.boxes = []
        
        data = configs.load_yaml(file)
        self.boxes.append(data["boundary"])
        self.boxes += data["obstacles"]

        for i, box in enumerate(self.boxes):
            assert len(box) == 4 and "wrong datasize at obstacle %d" % i


    def is_feasible(self, points):
        xl = points[:,0].min()
        xh = points[:,0].max()
        yl = points[:,1].min()
        yh = points[:,1].max()
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.boxes):
            is_outside = x_min >= xh or x_max <= xl or y_min >= yh or y_max <= yl
            #is_inside = x_min < xl and xh < x_max and y_min < yl and yh < y_max
            if i == 0:
                if is_outside:
                    return False
                
            elif not is_outside:
                return False
            
        return True
    
    def is_feasible_continuous(self, points):
        """
        Returns a continuous measure of feasibility with respect to obstacle avoidance.

        Parameters:
        points (np.array): Array of points representing parts of the quadrotor (e.g., tips).
        boxes (list of tuples): List of obstacle boxes, each defined as (x_min, y_min, x_max, y_max).

        Returns:
        float: A continuous measure of feasibility. Positive values indicate safe distances from obstacles,
            while negative values indicate violations of obstacle constraints.
        """
        min_distance = float('inf')  # Initialize with a large positive number

        for point in points:
            for box in self.boxes:
                x_min, y_min, x_max, y_max = box

                # Calculate distance from the point to the box edges
                dx = max(x_min - point[0], 0, point[0] - x_max)
                dy = max(y_min - point[1], 0, point[1] - y_max)

                # Euclidean distance to the nearest edge of the box
                distance_to_box = math.sqrt(dx**2 + dy**2)

                # Update the minimum distance to any obstacle
                min_distance = min(min_distance, distance_to_box)

        safety_threshold = 0.05

        # Feasibility measure: distance minus the threshold
        # Positive when safely away from obstacles, negative when too close or inside an obstacle
        return min_distance - safety_threshold


    def plot(self, ax: plt.Axes):
        lines = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.boxes):
            if i==0: #world boundary
                linewidth = 4
                lines += ax.plot([x_min, x_max], [y_min, y_min], 'orange', linewidth=linewidth)
                lines += ax.plot([x_min, x_max], [y_max, y_max], 'orange', linewidth=linewidth)
                lines += ax.plot([x_min, x_min], [y_min, y_max], 'orange', linewidth=linewidth)
                lines += ax.plot([x_max, x_max], [y_min, y_max], 'orange', linewidth=linewidth)
            else:
                w = x_max - x_min
                h = y_max - y_min
                line = ax.add_patch(Rectangle((x_min, y_min), w, h))
                lines.append(line)

        return lines

    
    def add_constraints(self, prog):
        pass


    def get_world(self):
        window = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.boxes):
            if i==0: #world boundary
                boundary = [x_min, y_min, x_max, y_max]
            else:
                window.append([x_min, y_min, x_max, y_max])
        return boundary, window