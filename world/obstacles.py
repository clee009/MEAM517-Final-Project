import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import configs
import pydrake.symbolic as sym


class Obstacles:
    def __init__(self, file: str, epsilon = 0.025):
        self.epsilon = epsilon
        
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.boxes = np.array(self.boxes)
        self.regions = self._convex_segmentation()
        n = len(self.regions)

        self.adj_boxes = []
        self.adj_areas = []
        self.adj_table = [n * [-1] for _ in range(n)]

        for i, (xi_min, yi_min, xi_max, yi_max) in enumerate(self.regions): 
            for j, (xj_min, yj_min, xj_max, yj_max) in enumerate(self.regions): 
                if i <= j:
                    continue

                x_min = max(xi_min, xj_min) - self.epsilon
                y_min = max(yi_min, yj_min) - self.epsilon
                x_max = min(xi_max, xj_max) + self.epsilon
                y_max = min(yi_max, yj_max) + self.epsilon

                if x_min < x_max and y_min < y_max:
                    area = (x_min - x_max) * (y_min - y_max)
                    if area > 2 * self.epsilon ** 2:
                        idx = len(self.adj_boxes)
                        self.adj_table[i][j] = self.adj_table[j][i] = idx
                        self.adj_areas.append(area)
                        self.adj_boxes.append((x_min, y_min, x_max, y_max))

    
    def get_region_ids(self, x):
        region_ids = []

        xe, ye = x[:2]
        eps = self.epsilon
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.regions):
            if (x_min - eps) < xe < (x_max + eps) and (y_min - eps) < ye < (y_max + eps):
                region_ids.append(i)

        return region_ids
    

    def _get_sample_space(self):
        x_min, y_min, x_max, y_max = self.boxes[0]
        sample_space = np.zeros((8,2))
        sample_space[:,0] = -np.pi/2
        sample_space[:,1] =  np.pi/2

        sample_space[0,0] = x_min
        sample_space[0,1] = x_max
        sample_space[1,0] = y_min
        sample_space[1,1] = y_max
        sample_space[4:,1] = np.array(self.vel_span)
        sample_space[4:,0] = -sample_space[4:,1]
        return sample_space
    

    def plot_trajectories(self, trajs: dict):
        fig = plt.figure(figsize=(8,6))

        ax = plt.axes()
        self.plot_obs(ax)

        for label, xx in trajs.items():
            ax.plot(xx[:,0], xx[:,1], '--', label=label)

        ax.axis("equal")
        plt.legend(loc='upper right')
        plt.savefig("./results/trajectories.svg")
        plt.show()


    def _convex_segmentation(self):
        x_lines = set()
        y_lines = set()
        for x_min, y_min, x_max, y_max in self.boxes:
            x_lines.add(x_min)
            x_lines.add(x_max)
            y_lines.add(y_min)
            y_lines.add(y_max)

        x_lines = sorted(list(x_lines))
        y_lines = sorted(list(y_lines))

        regions = []
        for x_min, x_max in zip(x_lines, x_lines[1:]):
            for y_min, y_max in zip(y_lines, y_lines[1:]):
                centroid = np.array([[x_min + x_max, y_min + y_max]]) / 2
                if self.is_feasible(centroid):
                    regions.append([x_min, y_min, x_max, y_max])

        return regions
    

    def is_state_feasible(self, state):
        x, y = state[:2]
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.boxes):
            inside = x_min < x < x_max and y_min < y < y_max
            if i == 0:
                if not inside:
                    return False
                
            elif inside:
                return False
            
        return True


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
                dx = np.max(x_min - point[0], 0, point[0] - x_max)
                dy = np.max(y_min - point[1], 0, point[1] - y_max)

                # Euclidean distance to the nearest edge of the box
                distance_to_box = np.sqrt(dx**2 + dy**2)

                # Update the minimum distance to any obstacle
                min_distance = np.min(min_distance, distance_to_box)

        # Define a threshold distance under which we consider the point to be too close
        safety_threshold = 0.05  # This can be adjusted based on the specific requirements

        # Feasibility measure: distance minus the threshold
        # Positive when safely away from obstacles, negative when too close or inside an obstacle
        return min_distance - safety_threshold


    def plot_obs(self, ax: plt.Axes, plot_segs=False):
        lines = []
        boxes = self.regions if plot_segs else self.boxes
        for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            if i==0: #world boundary
                linewidth = 4
                lines += ax.plot([x_min, x_max], [y_min, y_min], 'orange', linewidth=linewidth)
                lines += ax.plot([x_min, x_max], [y_max, y_max], 'orange', linewidth=linewidth)
                lines += ax.plot([x_min, x_min], [y_min, y_max], 'orange', linewidth=linewidth)
                lines += ax.plot([x_max, x_max], [y_min, y_max], 'orange', linewidth=linewidth)
            else:
                w = x_max - x_min
                h = y_max - y_min
                line = ax.add_patch(Rectangle((x_min, y_min), w, h, alpha = i / len(boxes)))
                lines.append(line)

        return lines


    def get_world(self):
        window = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.boxes):
            if i==0: #world boundary
                boundary = [x_min, y_min, x_max, y_max]
            else:
                window.append([x_min, y_min, x_max, y_max])
        return boundary, window