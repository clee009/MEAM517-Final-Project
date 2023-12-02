import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from obstacles import Obstacles
from quadrotor_with_pendulum import QuadrotorPendulum


class Animation:
    def __init__(self, obstacles: Obstacles, quad: QuadrotorPendulum, tf = 10, num_frames = 60):
        self.obstacles = obstacles

        self.traj = np.zeros((1,8))
        self.quad = quad
        self.tf = tf
        self.num_frames = num_frames

        self.drone_width = quad.lb
        self.pendu_width = quad.l1
    
    def set_trajectory(self, traj):
        self.traj = traj


    def plot_trajectory(self, i, ax: plt.Axes):
        return ax.plot(self.traj[:i+1,0], self.traj[:i+1,1], '--', label='actual trajectory')


    def plot_quadrotor(self, state, ax: plt.Axes):
        x, y, theta, phi = state[:4]
        x_radi = self.quad.lb * np.cos(theta) / 2
        y_radi = self.quad.lb * np.sin(theta) / 2

        lines = []
        lines += ax.plot([x + x_radi, x - x_radi], [y + y_radi, y - y_radi], 'g')
        lines += ax.plot([x, x + self.quad.l1 * np.sin(phi)], [y, y - self.quad.l1 * np.cos(phi)], 'r')
        
        ends = self.quad.get_ends(state)
        lines += ax.plot(ends[:,0], ends[:,1], 'ro') # Your original list

        return lines


    def animate(self):
        rc('animation', html='jshtml')

        fig = plt.figure(figsize=(8,6))
        ax = plt.axes()

        x_max = self.traj[:,0].max()
        x_min = self.traj[:,0].min()
        y_max = self.traj[:,1].max()
        y_min = self.traj[:,1].min()

        frame_ids = np.linspace(0, len(self.traj) - 1, self.num_frames)
        frame_ids = [int(np.round(x)) for x in frame_ids]
        anim_states = np.zeros((self.num_frames, 8))
        for i, frame_id in enumerate(frame_ids):
            anim_states[i,:] = self.traj[frame_id,:]

        x_padding = 0.25 * (x_max - x_min)
        y_padding = 0.25 * (y_max - y_min)

        def frame(i):
            ax.clear()

            lines = []
            lines += self.obstacles.plot(ax)
            lines += self.plot_trajectory(frame_ids[i], ax)
            lines += self.plot_quadrotor(anim_states[i], ax)
            
            if(np.abs((x_max - x_min) - (y_max - y_min)) < 5):
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

            ax.set_xlabel('y (m)')
            ax.set_ylabel('z (m)')
            ax.set_aspect('equal')
            ax.legend(loc='upper left')

            return lines

        return animation.FuncAnimation(fig, frame, frames=self.num_frames, blit=False, repeat=False), fig