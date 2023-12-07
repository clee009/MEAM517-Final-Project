import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import configs


class Obstacles:
    def __init__(self, file: str):
        self.boxes = []

        
        data = configs.load_yaml(file)
        self.boxes.append(data["boundary"])
        self.boxes += data["obstacles"]

        for i, box in enumerate(self.boxes):
            assert len(box) == 4 and "wrong datasize at obstacle %d" % i

        self.convex_region_segmentation()


    def convex_region_segmentation(self):
        self.segments = [self.boxes[0]]
        x_lines = set()
        y_lines = set()
        for x_min, y_min, x_max, y_max in self.boxes:
            x_lines.add(x_min)
            x_lines.add(x_max)
            y_lines.add(y_min)
            y_lines.add(y_max)

        x_lines = sorted(list(x_lines))
        y_lines = sorted(list(y_lines))

        for x_min, x_max in zip(x_lines, x_lines[1:]):
            for y_min, y_max in zip(y_lines, y_lines[1:]):
                box = [x_min, y_min, x_max, y_max]
                if box not in self.boxes:
                    self.segments.append(box)



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
        boxes = self.segments if plot_segs else self.boxes
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