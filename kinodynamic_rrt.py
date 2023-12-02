import numpy as np
import numpy.linalg as npl
import lqrrt
import configs
from quadrotor_with_pendulum import QuadrotorPendulum
from obstacles import Obstacles
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
        A, B = self.quad.GetLinearizedDynamics(u, x)

        S = np.identity(len(self.Q)) #todo : implement ricatti
        K = -self.iR @ B.T @ S

        return S, K
    
    
    def erf(self, xgoal, x):
        """
        Returns error e given two states xgoal and x.

        """
        return xgoal - x
    

    def is_feasible(self, x, u):
        #if (u > self.quad.input_max).any() or (u < self.quad.input_min).any():
            #return False
        
        ends = self.quad.get_ends(x)
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.obs.boxes):
            for xe, ye in ends:
                if i==0: 
                    if not (x_min <= xe <= x_max and y_min <= ye <= y_max): 
                        return False
                    
                elif x_min <= xe <= x_max or y_min <= ye <= y_max:
                    return False

        return True
    

    def dynamics(self, x, u, dt):
        def f(_, x):
            return self.quad.evaluate_f(u, x)
        
        sol = solve_ivp(f, (0, dt), x, first_step=dt)
        return sol.y[:,-1].ravel()
    

    def find_trajectory(self, x0, goal):
        buff = 1
        x_min, y_min, x_max, y_max = self.obs.boxes[0]
        sample_space = [(x_min, x_max),
                        (y_min, y_max),
                        (-np.pi, np.pi),
                        (-np.pi, np.pi),
                        (-buff, buff),
                        (-buff, buff),
                        (-np.pi, np.pi),
                        (-np.pi, np.pi),]

        xrand_gen = None

        ################################################# PLAN

        goal_buffer = 8 * [0.05]
        constraints = lqrrt.Constraints(nstates=len(self.Q), ncontrols=len(self.R),
                                        goal_buffer= goal_buffer, is_feasible=self.is_feasible)

        planner = lqrrt.Planner(self.dynamics, self.lqr, constraints,
                                horizon=5, dt=self.dt,
                                error_tol=np.copy(goal_buffer)/2, erf=self.erf,
                                min_time=0, max_time=10, max_nodes=1E5,
                                goal0=goal, printing=True)

        planner.update_plan(x0, sample_space, xrand_gen=xrand_gen, finish_on_goal=False)
        print(planner.get_state)