import numpy as np
import numpy.linalg as npl
import lqrrt
import configs
from quadrotor_with_pendulum import QuadrotorPendulum
from obstacles import Obstacles
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp


class KinodtnamicRRT:
    def __init__(self, file: str, obs: Obstacles, quad: QuadrotorPendulum):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.R = np.array(self.R)
        self.iR = np.linalg.inv(self.R)

        self.Q = np.array(self.Q)
        self.Qf = np.array(self.Qf)

        self.obs = obs
        self.quad = quad


    def lqr(self, x, u): #lqrrt input wrapping
        A, B = self.quad.GetLinearizedDynamics(self.quad.u_d(), x)
        S = solve_continuous_are(A, B, self.quad.Q, self.quad.R)
        K = -np.linalg.inv(self.quad.R) @ B.T @ S
        return S, K
    
    
    def erf(self, xgoal, x):
        """
        Returns error e given two states xgoal and x.

        """
        e = x - xgoal
        return e
    

    def is_feasible(self, x, u):
        if (u > self.quad.input_max).any() or (u < self.quad.input_min).any():
            return False
        
        ends = self.quad.get_ends(x)
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
    

    def find_trajectory(self, x0, goal):
        buff = 0.1
        x_min, y_min, x_max, y_max = self.obs.boxes[0]
        sample_space = [(x_min, x_max),
                        (y_min, y_max),
                        (-np.pi/2, np.pi/2),
                        (-np.pi/2, np.pi/2),
                        (-buff, buff),
                        (-buff, buff),
                        (-buff, buff),
                        (-buff, buff),]

        xrand_gen = None

        ################################################# PLAN

        goal_buffer = 8 * [0.1]
        constraints = lqrrt.Constraints(nstates=len(self.Q), ncontrols=len(self.R),
                                        goal_buffer=goal_buffer, is_feasible=self.is_feasible)

        planner = lqrrt.Planner(self.dynamics, self.lqr, constraints,
                                horizon=30, dt=self.dt, erf=self.erf,
                                min_time=0, max_time=30, max_nodes=1e5,
                                goal0=goal, printing=True)

        planner.update_plan(x0, sample_space, goal_bias=0.2, xrand_gen=xrand_gen, finish_on_goal=False, u_d=self.quad.u_d())
        
        import matplotlib.pyplot as plt
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

        dt = 0.03  # s
        T = planner.T  # s
        t_arr = np.arange(0, T, dt)

        # Preallocate results memory
        x = np.copy(x0)
        x_history = np.zeros((len(t_arr), 8))
        goal_history = np.zeros((len(t_arr), 8))
        u_history = np.zeros((len(t_arr), 2))

        # Interpolate plan
        for i, t in enumerate(t_arr):

            # Planner's decision
            x = planner.get_state(t)
            u = planner.get_effort(t)

            # Record this instant
            x_history[i, :] = x
            goal_history[i, :] = goal
            u_history[i, :] = u

        return t_arr, x_history, u_history