import numpy as np
import lqrrt
import configs
from quadrotor import QuadrotorPendulum
from world import SignedDistanceField
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random


class PathPlannerLQRRT:
    def __init__(self, file: str, quad: QuadrotorPendulum, obs: SignedDistanceField):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.Q = np.array(quad.Q)
        self.R = np.array(quad.R)
        self.iR = np.linalg.inv(quad.R)

        self.obs = obs
        self.quad = quad

        sample_space = self._get_sampling_space()
        self.sample_means = np.mean(sample_space, axis=1)
        self.sample_spans = np.diff(sample_space).flatten()
        self.goal_bias = 8 * [self.goal_bias]

        self.x0 = np.array(self.x0)
        self.xf = np.array(self.xf)


    def record_state(self, x):
        for i in self.obs.get_region_ids(x):
            self.accessible[i] = True


    def _get_sampling_space(self):
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
        return sample_space


    def lqr(self, x, u): #lqrrt input wrapping
        A, B = self.quad.GetLinearizedDynamics(x, self.quad.u_f)
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -self.iR @ B.T @ S
        return S, K
    
    
    def erf(self, xgoal, x):
        return x - xgoal
    

    def is_feasible(self, x, u):
        if (u > self.quad.input_max).any() or (u < self.quad.input_min).any():
            return False
            
        xe, ye = x[:2]
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.obs.boxes):
            is_inside = x_min < xe < x_max and y_min < ye < y_max
            if i==0: 
                if not is_inside: 
                    return False
                    
            elif is_inside:
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
    

    def xrand_gen(self, planner):
        while True:
            adj_ids = set() #find all adj
            for i, accessible in enumerate(self.accessible):
                if not accessible:
                    continue

                for j, adj_id in enumerate(self.obs.adj_table[i]):
                    if adj_id != -1 and not self.accessible[j]:
                        adj_ids.add(adj_id)

            if not adj_ids:
                return self.xf 
            
            adj_ids = list(adj_ids)

            weights = [self.obs.adj_areas[idx] for idx in adj_ids]
            alpha = 1. / sum(weights)
            weights = [w * alpha for w in weights]

            adj_id = random.choices(population = adj_ids, weights = weights)[0]
            x_min, y_min, x_max, y_max = self.obs.adj_boxes[adj_id]

            xrand = self.sample_means + self.sample_spans * (np.random.sample(8)-0.5)
            xrand[0] = np.random.uniform(x_min, x_max)
            xrand[1] = np.random.uniform(y_min, y_max)
            for i, choice in enumerate(np.greater(self.goal_bias, np.random.sample())):
                if choice:
                    xrand[i] = self.xf[i]
            
            if self.is_feasible(xrand, np.zeros(2)):
                return xrand
                    
        return xrand
    

    def get_planner(self):
        self.accessible = len(self.obs.regions) * [False]
        self.record_state(self.x0)

        ################################################# PLAN

        goal_buffer = 8 * [self.goal_buffer]
        constraints = lqrrt.Constraints(nstates=len(self.Q), ncontrols=len(self.R),
                                        goal_buffer=goal_buffer, is_feasible=self.is_feasible)

        planner = lqrrt.Planner(self.dynamics, self.lqr, constraints, error_tol=self.epsilon, 
                                horizon=self.horizon, dt=self.dt, erf=self.erf, 
                                min_time=0, max_time=self.max_time, max_nodes=self.max_node,
                                goal0=self.xf, printing=True)
        
        xrand_gen = self.xrand_gen if self.use_segment else None
        record_state = self.record_state if self.use_segment else None

        planner.update_plan(self.x0, self._get_sampling_space(), goal_bias=self.goal_bias, record_state=record_state, 
                            xrand_gen=xrand_gen, finish_on_goal=False, u_d=self.quad.u_f)
        return planner