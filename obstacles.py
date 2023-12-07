import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import configs


class Obstacles:
    def __init__(self, file: str):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.regions = self._convex_segmentation()

    
    def get_region_ids(self, x):
        region_ids = []

        xe, ye = x[:2]
        eps = self.epsilon
        for i, (x_min, y_min, x_max, y_max) in self.regions:
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
                box = [x_min, y_min, x_max, y_max]
                if box not in self.boxes:
                    regions.append(box)

        return regions


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


    def plot(self, ax: plt.Axes, plot_segs=False):
        lines = []
        boxes = self.regions if plot_segs else self.boxes
        colors = ['k','b','y','g','r']
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
                idx = i % len(colors)
                line = ax.add_patch(Rectangle((x_min, y_min), w, h, facecolor=colors[idx]))
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