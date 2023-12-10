import numpy as np
from scipy.signal import cont2discrete
from typing import List, Tuple
import torch
from torch.autograd.functional import hessian, jacobian
from scipy.integrate import solve_ivp
from matplotlib import rc
import matplotlib.animation as animation

from lqrrt import PathPlannerLQRRT

import configs
from ilqr import iLQR
import matplotlib.pyplot as plt


class iiLQR(iLQR):
    def __init__(self, file: str, rrt: PathPlannerLQRRT):
        super().__init__(file, rrt)
        self.traj_hist = []


    
    def visualize_trajectory(self, xx):
        print(len(xx))

        fig = plt.figure()
        ax = fig.add_subplot()

        self.sdf.plot_obs(ax)
        xx = np.array([xk[:2] for xk in xx])

        ax.plot(xx[:,0], xx[:,1], '--', label='actual trajectory')
        ax.axis('equal')
        plt.show()


    def animate(self):
        rc('animation', html='jshtml')
        fig = plt.figure(figsize=(8,6))
        ax = plt.axes()

        self.num_frames = 300
        frame_ids = np.linspace(0, len(self.traj_hist) - 1, self.num_frames)
        frame_ids = [int(np.round(x)) for x in frame_ids]
        traj_hist = [self.traj_hist[i] for i in frame_ids]

        x_min, y_min, x_max, y_max = self.sdf.boxes[0]
        def frame(i):
            ax.clear()

            lines = []
            lines += self.sdf.plot_obs(ax)
            xx = traj_hist[i]
            lines += ax.plot(xx[:,0], xx[:,1], '--', label='actual trajectory')

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel('y (m)')
            ax.set_ylabel('z (m)')
            ax.set_aspect('equal')
            ax.legend(loc='upper left')

            return lines
        
        anim = animation.FuncAnimation(fig, frame, frames=self.num_frames, blit=False, repeat=False)
        plt.close()
        return anim
        

    def calculate_optimal_trajectory(self, x0, uu_guess: list, dt):
        uu_traj = []
        xx_traj = [np.copy(x0)]

        xx_guess = [np.copy(x0)]
        for uk in uu_guess:
            xx_guess.append(self.dynamics(xx_guess[-1], uk))
        
        while len(uu_traj) < len(uu_guess):
            k = len(uu_traj)
            x0 = xx_traj[-1]

            goal_idx = len(uu_guess)
            while True:
                uu = uu_guess[k:goal_idx].copy()
                xf = xx_guess[goal_idx]
                bump_idx, xx, uu = super().calculate_optimal_trajectory(x0, xf, uu, dt)
                print(k, bump_idx)
                if bump_idx == -1:
                    assert (xx_traj[-1] == xx[0]).all()
                    uu_traj += uu
                    xx_traj += xx[1:]
                    break

                goal_idx = k + bump_idx - 1


            #print(goal_idx, len(uu_traj))
            traj = xx_traj + xx_guess[len(xx_traj):]
            self.traj_hist.append(np.array(traj))
            self.visualize_trajectory(traj)

        return xx_traj, uu_traj
