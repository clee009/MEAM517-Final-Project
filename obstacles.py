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


    def is_feasible(self, points):
        xl = points[:,0].min()
        xh = points[:,0].max()
        yl = points[:,1].min()
        yh = points[:,1].max()
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.boxes):
            is_outside = (x_min >= xh or x_max <= xl) and (y_min >= yh or y_max <= yl)
            #is_inside = x_min < xl and xh < x_max and y_min < yl and yh < y_max
            if i == 0:
                if is_outside:
                    return False
                
            elif not is_outside:
                return False
            
        return True


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