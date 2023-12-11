import numpy as np
from scipy.signal import cont2discrete
from typing import List, Tuple
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from lqrrt import PathPlannerLQRRT
import matplotlib.pyplot as plt

import configs
from tqdm import tqdm


class iLQR:
    def __init__(self, file: str, rrt: PathPlannerLQRRT):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.Q = rrt.Q
        self.R = rrt.R
        self.Qf = rrt.quad.Qf

        self.sdf = rrt.obs
        self.quad = rrt.quad
        self.uf = rrt.quad.u_f
        self.xf = rrt.xf
        self.dt = rrt.dt
        self.eps = rrt.epsilon
        

    def total_cost(self, xx, uu):
        cost = sum(self.running_cost(*args) for args in zip(xx, uu, xx[1:]))
        return cost + self.terminal_cost(xx[-1])
    

    def get_linearized_discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A, B = self.quad.GetLinearizedDynamics(x, u)
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0],))
        [Ad, Bd, _, _, _] = cont2discrete((A, B, C, D), self.dt)
        return Ad, Bd
    

    def calc_barrier_cost(self, xk):
        sdf = self.sdf.calc_sdf(xk)
        if sdf >= 0:
            return 0
        
        return self.q1 * sdf ** 2
        #return self.q1 * np.exp(self.q2 * self.sdf.calc_sdf(xk))
         

    def running_cost(self, xk, uk, xd) -> float:
        x_cost = (xk - xd).T @ self.Q @ (xk - xd)
        u_cost = (uk - self.uf).T @ self.R @ (uk - self.uf)
        b_cost = self.calc_barrier_cost(xk)
        return 0.5 * (x_cost + u_cost) + b_cost
    

    def grad_running_cost(self, xk, uk, xd) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        """
        grad = np.zeros((10,))

        #TODO: Compute the gradient
        grad[:8] = (xk - xd).T @ self.Q
        grad[8:] = (uk - self.uf).T @ self.R

        #c = self.q2 * self.calc_barrier_cost(xk)
        sdf = self.sdf.calc_sdf(xk)
        if sdf < 0:
            grad[:2] += 2 * self.q1 * sdf * self.sdf.calc_grad(xk)

        return grad 
    

    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        H = np.zeros((10, 10))

        # TODO: Compute the hessian
        H[:8,:8] = self.Q
        H[8:,8:] = self.R

        #c = self.q2**2 * self.calc_barrier_cost(xk)
        #grad = self.sdf.calc_grad(xk)

        sdf = self.sdf.calc_sdf(xk)
        if sdf < 0:
            grad = self.sdf.calc_grad(xk)
            H[:2,:2] += 2* self.q1 * grad @ grad.T

        return H
    

    def terminal_cost(self, xf: np.ndarray) -> float:
        return 0.5 * (xf - self.xf).T @ self.Qf @ (xf - self.xf)
    

    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        grad = np.zeros((8))

        # TODO: Compute the gradient
        grad = (xf - self.xf).T @ self.Qf
        return grad
    
        
    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        return self.Qf
    

    def dynamics(self, xc, uc):
        def f(_, x):
            return self.quad.evaluate_f(uc, x)
        
        sol = solve_ivp(f, (0, self.dt), xc, first_step=self.dt)
        return sol.y[:,-1].ravel()
    

    def forward_pass(self, xx: List[np.ndarray], uu: List[np.ndarray], dd: List[np.ndarray], KK: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: list of states, should be length N
        :param uu: list of inputs, should be length N-1
        :param dd: list of "feed-forward" components of iLQR update, should be length N-1
        :param KK: list of "Feedback" LQR gain components of iLQR update, should be length N-1
        :return: A tuple (xx, uu) containing the updated state and input
                 trajectories after applying the iLQR forward pass
        """
        
        xtraj = [np.zeros((8,))] * self.N
        utraj = [np.zeros((2,))] * (self.N - 1)

        xtraj[0] = xx[0]
        for k in range(self.N-1):
            delta_xk = xtraj[k] - xx[k]
            du = KK[k] @ delta_xk + self.alpha * dd[k]
            utraj[k] = np.clip(uu[k] + du, self.quad.input_min, self.quad.input_max)
            xtraj[k+1] = self.dynamics(xtraj[k], utraj[k])
          

        # TODO: compute forward pass
        return xtraj, utraj
    

    def backward_pass(self, xx, uu, xx_resamp=[], damping=0):
        dd = [np.zeros((2,))] * (self.N - 1)
        KK = [np.zeros((2, 8))] * (self.N - 1)
        H_last = self.hess_terminal_cost(xx[-1])
        g_last = self.grad_terminal_cost(xx[-1])

        for k in range(self.N-2,-1,-1):
            Ak, Bk = self.get_linearized_discrete_dynamics(xx[k], uu[k])
            Hl = self.hess_running_cost(xx[k], uu[k])

            if not xx_resamp:
                gl = self.grad_running_cost(xx[k], uu[k], xx[k+1])
            else:
                dist = [np.linalg.norm(x - xx[k]) for x in xx_resamp]
                xd = xx_resamp[np.argmin(dist)]
                gl = self.grad_running_cost(xx[k], uu[k], xd)
            
            Qx = gl[:8] + Ak.T @ g_last
            Qu = gl[8:] + Bk.T @ g_last
            Qxx = Hl[:8,:8] + Ak.T @ H_last @ Ak
            Quu = Hl[8:,8:] + Bk.T @ H_last @ Bk
            Qux = Hl[8:,:8] + Bk.T @ H_last @ Ak

            evals, evecs = np.linalg.eig(Quu)
            evals = np.maximum(0, evals) + damping
            Quu = evecs @ np.diag(evals) @ evecs.T
            
            KK[k] = -np.linalg.solve(Quu, Qux)
            dd[k] = -np.linalg.solve(Quu, Qu)
            H_last = Qxx - KK[k].T @ Quu @ KK[k]
            g_last = Qx  - KK[k].T @ Quu @ dd[k]

        # TODO: compute backward pass

        return dd, KK
    

    def visualize(self, xx, i):
        print(len(xx))

        fig = plt.figure()
        ax = fig.add_subplot()
        self.sdf.plot_obs(ax)
        xx = np.array([xk[:2] for xk in xx])

        ax.plot(xx[:,0], xx[:,1], '--', label='actual trajectory')
        ax.axis('equal')
        ax.set_title("iteration: %d" % i)
        plt.show()


    def backward_pass_flatten(self, xx, uu, xx_resamp):
        dd = [np.zeros((2,))] * (self.N - 1)
        KK = [np.zeros((2, 8))] * (self.N - 1)
        H_last = self.hess_terminal_cost(xx[-1])
        g_last = self.grad_terminal_cost(xx[-1])

        for k in range(self.N-2,-1,-1):
            Ak, Bk = self.get_linearized_discrete_dynamics(xx[k], uu[k])
            Hl = self.hess_running_cost(xx[k], uu[k])

            dist = [np.linalg.norm(x - xx[k]) for x in xx_resamp]
            xd = xx_resamp[np.argmin(dist)]
            gl = self.grad_running_cost(xx[k], uu[k], xd)
            
            Qx = gl[:8] + Ak.T @ g_last
            Qu = gl[8:] + Bk.T @ g_last
            Qxx = Hl[:8,:8] + Ak.T @ H_last @ Ak
            Quu = Hl[8:,8:] + Bk.T @ H_last @ Bk
            Qux = Hl[8:,:8] + Bk.T @ H_last @ Ak

            evals, evecs = np.linalg.eig(Quu)
            evals = np.maximum(0, evals) + 1e-3
            Quu = evecs @ np.diag(evals) @ evecs.T
            
            KK[k] = -np.linalg.solve(Quu, Qux)
            dd[k] = -np.linalg.solve(Quu, Qu)
            H_last = Qxx - KK[k].T @ Quu @ KK[k]
            g_last = Qx  - KK[k].T @ Quu @ dd[k]

        # TODO: compute backward pass

        return dd, KK
    
    
    def flatten_trajectory(self, xx, uu):
        eps = self.flatten_steps * self.eps
        xx_resamp = [np.copy(xx[0])]
        for xk in xx:
            if not (np.abs(xx_resamp[-1] - xk) <= eps).all():
                xx_resamp.append(xk)

        xx_resamp.append(xx[-1])

        J = np.inf
        for _ in range(self.flatten_iters):
            dd, KK = self.backward_pass_flatten(xx, uu, xx_resamp)
            xx, uu = self.forward_pass(xx, uu, dd, KK)
            J_temp = self.total_cost(xx, uu)
            if np.abs(J - J_temp) < 1e-3:
                break

            J = J_temp

        print("down-length: ", len(xx_resamp))
        return xx, uu

    
    def calc_lqr_cost(self, x0, xf, max_cost):
        xk = np.copy(x0)

        cost = 0
        while not (np.abs(xk - xf) <= self.eps).all():
            uk = self.quad.compute_lqr_feedback(xk, x_goal=xf)
            uk = np.clip(uk, self.quad.input_min, self.quad.input_max)
            xd = self.dynamics(xk, uk)
            if not self.sdf.is_state_feasible(xd):
                return np.inf
                
            cost += self.running_cost(xk, uk, xd)
            if cost >= max_cost:
                return np.inf

            xk = xd

        return cost
            


    def calculate_optimal_trajectory(self, x0, xf, uu_guess, dt):
        self.xf = xf
        self.dt = dt
        self.N = len(uu_guess) + 1

        # Get an initial, dynamically consistent guess for xx by simulating the quadrotor
        xx = [x0]
        for k in range(self.N-1):
            xx.append(self.dynamics(xx[k], uu_guess[k]))

        J = self.total_cost(xx, uu_guess)
        uu = uu_guess

        for i in range(100):
            if self.enable_visualization:
                self.visualize(xx, i)
                
            xx_resamp = []
            if i % self.flatten_period == 0: # flatten!
                eps = self.flatten_steps * self.eps
                xx_resamp.append(np.copy(xx[0]))
                for xk in xx:
                    if not (np.abs(xx_resamp[-1] - xk) <= eps).all():
                        xx_resamp.append(xk)

                xx_resamp.append(xf)

            dd, KK = self.backward_pass(xx, uu, xx_resamp)
            xx, uu = self.forward_pass(xx, uu, dd, KK)
            J_temp = self.total_cost(xx, uu)
            if abs(J_temp-J) < 1e-3:
                break

            J = J_temp
            print("[iter-{}]\tJ: {:.3f}".format(i, J))

        print(f'Converged to cost {J}')
        return xx, uu
