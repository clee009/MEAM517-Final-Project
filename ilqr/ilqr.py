import pydrake.all as drake


import numpy as np
from world.obstacles import Obstacles
from quadrotor_with_pendulum import QuadrotorPendulum


class iLQR:
    def __init__(self, quad: QuadrotorPendulum, obs: Obstacles):
        self.quad = quad
        self.obs = obs

        self.Q = quad.Q
        self.R = quad.R
        self.iR = np.linalg.inv(quad.R)
        self.Qf = quad.Qf
        

    
    def update_state_trajectory(self, traj):
        self.traj = traj


    
    def set_init_guess(self, x_traj, u_traj, dt):
        self.dt = dt
        self.N = len(x_traj)
        self.x_traj = x_traj
        self.u_traj = u_traj

        self.context = drake.MathematicalProgram()



    def running_cost(self, xk, uk):
        region_ids = self.obs.get_region_ids(xk)


    def total_cost(self):
