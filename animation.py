import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from world import Obstacles
from quadrotor_with_pendulum import QuadrotorPendulum


class Animation:
    RRT=0
    ILQR=1
    def __init__(self, quad: QuadrotorPendulum, obs: Obstacles, num_frames = 60):
        self.obs = obs
        self.traj = np.zeros((1,8))
        self.quad = quad
        self.num_frames = num_frames
        

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


    def animate(self, title=""):
        rc('animation', html='jshtml')

        fig = plt.figure(figsize=(8,6))
        ax = plt.axes()

        frame_ids = np.linspace(0, len(self.traj) - 1, self.num_frames)
        frame_ids = [int(np.round(x)) for x in frame_ids]
        anim_states = np.zeros((self.num_frames, 8))
        for i, frame_id in enumerate(frame_ids):
            anim_states[i,:] = self.traj[frame_id,:]

        x_min, y_min, x_max, y_max = self.obs.boxes[0]

        def frame(i):
            ax.clear()

            lines = []
            lines += self.obs.plot_obs(ax)
            lines += self.plot_trajectory(frame_ids[i], ax)
            lines += self.plot_quadrotor(anim_states[i], ax)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel('y (m)')
            ax.set_ylabel('z (m)')
            ax.set_aspect('equal')
            ax.legend(loc='upper left')
            if title:
                ax.set_title(title)

            return lines
        
        anim = animation.FuncAnimation(fig, frame, frames=self.num_frames, blit=False, repeat=False)
        plt.close()
        return anim
    

def create_animation(suffix: str, mode = Animation.RRT):
    R = np.eye(2)
    Q = np.diag([10, 10, 1, 1, 1, 1, 1, 1])
    Qf = Q

    # End time of the simulation
    xf = np.array([6, 3, 0, 0, 0, 0, 0, 0])
    quad = QuadrotorPendulum(Q, R, Qf, xf, input_max=30)
    obs = Obstacles("./configs/world%s.yaml" % suffix)

    file = "./initial_guesses" if mode == Animation.RRT else "./results"
    file = os.path.join(file, "x%s.npy" % suffix)

    anime = Animation(quad, obs)
    anime.set_trajectory(np.load(file))
    return anime