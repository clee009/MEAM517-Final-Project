import numpy as np
import lqrrt
import configs
from quadrotor_with_pendulum import QuadrotorPendulum
from obstacles import Obstacles
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class PathPlannerLQRRT:
    def __init__(self, file: str, obs: Obstacles, quad: QuadrotorPendulum):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.Q = np.array(quad.Q)
        self.R = np.array(quad.R)
        self.iR = np.linalg.inv(quad.R)

        self.obs = obs
        self.quad = quad


    def lqr(self, x, u): #lqrrt input wrapping
        A, B = self.quad.GetLinearizedDynamics(self.quad.u_d(), x)
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -self.iR @ B.T @ S
        return S, K
    
    
    def erf(self, xgoal, x):
        return x - xgoal
    

    def is_feasible(self, x, u):
        if (u > self.quad.input_max).any() or (u < self.quad.input_min).any():
            return False
            
        ends = self.quad.get_ends(x)
        return self.obs.is_feasible(ends)
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.obs.boxes):
            for xe, ye in ends:
                if i==0: 
                    if not (x_min <= xe <= x_max and y_min <= ye <= y_max): 
                        return False
                    
                elif x_min <= xe <= x_max and y_min <= ye <= y_max:
                    return False

        return True
    

    def dynamics(self, xc, uc, dt):
        def f(_, x):
            return self.quad.evaluate_f(uc, x)
        
        sol = solve_ivp(f, (0, dt), xc, first_step=dt)
        return sol.y[:,-1].ravel()
    

    def plot_result(self, planner: lqrrt.Planner):
        x_min, y_min, x_max, y_max = self.obs.boxes[0]

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        self.obs.plot(plt.gca())

        plt.scatter(planner.tree.state[:,0], planner.tree.state[:,1])
        for ID in range(planner.tree.size):
            x_seq = np.array(planner.tree.x_seq[ID])
            if ID in planner.node_seq:
                plt.plot((x_seq[:,0]), (x_seq[:,1]), color='r', zorder=2)
            else:
                plt.plot((x_seq[:,0]), (x_seq[:,1]), color='0.75', zorder=1)
        plt.show()


    def interpolate_trajectory(self, planner: lqrrt.Planner):
        T = planner.T  # s
        t_arr = np.arange(0, T, self.traj_dt)

        # Preallocate results memory
        x_history = np.zeros((len(t_arr), 8))
        u_history = np.zeros((len(t_arr), 2))

        # Interpolate plan
        for i, t in enumerate(t_arr):
            x_history[i, :] = planner.get_state(t)
            u_history[i, :] = planner.get_effort(t)

        return t_arr, x_history, u_history
    

    def get_planner(self, x0, goal):
        x_min, y_min, x_max, y_max = self.obs.boxes[0]
        sample_space = np.zeros((8,2))
        sample_space[:,0] = -np.pi/2
        sample_space[:,1] =  np.pi/2

        sample_space[0,0] = x_min
        sample_space[0,1] = x_max
        sample_space[1,0] = y_min
        sample_space[1,1] = y_max
        sample_space[4:,1] = np.array(self.vel_span)
        sample_space[4:,0] = -sample_space[4:,1]

        xrand_gen = None

        ################################################# PLAN

        goal_buffer = 8 * [self.goal_buffer]
        constraints = lqrrt.Constraints(nstates=len(self.Q), ncontrols=len(self.R),
                                        goal_buffer=goal_buffer, is_feasible=self.is_feasible)

        planner = lqrrt.Planner(self.dynamics, self.lqr, constraints, error_tol=self.epsilon, 
                                horizon=self.horizon, dt=self.dt, erf=self.erf, 
                                min_time=0, max_time=self.max_time, max_nodes=self.max_node,
                                goal0=goal, printing=True)
        planner.update_plan(x0, sample_space, goal_bias=self.goal_bias, xrand_gen=xrand_gen, finish_on_goal=False, u_d=self.quad.u_d())
        return planner