import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import configs


class World:
    def __init__(self, file: str):
        configs.load_yaml(file)
        self.init_state = np.array(self.init_state)
        self.goal_state = np.array(self.goal_state)


    def plot(self, ax: plt.Axes, lines: list):
        x_min, y_min, x_max, y_max = self.boundary
        lines += ax.plot([x_min, x_max], [y_min, y_min], 'orange', linewidth=4)
        lines += ax.plot([x_min, x_max], [y_max, y_max], 'orange', linewidth=4)
        lines += ax.plot([x_min, x_min], [y_min, y_max], 'orange', linewidth=4)
        lines += ax.plot([x_max, x_max], [y_min, y_max], 'orange', linewidth=4)

        for x_min, y_min, x_max, y_max in self.get_obstacles():
            w = x_max - x_min
            h = y_max - y_min
            line = ax.add_patch(Rectangle((x_min, y_min), w, h))
            lines.append(line)

    
    def add_constraints(self, prog):
        pass