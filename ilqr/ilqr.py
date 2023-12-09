import numpy as np
from scipy.signal import cont2discrete
from typing import List, Tuple
import torch
from torch.autograd.functional import hessian, jacobian
from scipy.integrate import solve_ivp

from lqrrt import PathPlannerLQRRT

import configs


class iLQR:
    def __init__(self, file: str, rrt: PathPlannerLQRRT):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.Q = rrt.Q
        self.R = rrt.R
        self.Qf = rrt.quad.Qf

        self.sdf = rrt.obs
        self.quad = rrt.quad

        self.x0 = rrt.x0
        self.xf = rrt.xf
        self.uf = rrt.quad.u_f
        

    def total_cost(self, xx, uu):
        cost = 0
        with torch.no_grad():
            for xk, uk in zip(xx, uu):
                cost += self.running_cost(xk, uk)
                cost += self.sdf.barrier_func(xk).float()

        return cost + self.terminal_cost(xx[-1])
    

    def get_linearized_discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: state
        :param u: input
        :return: the discrete linearized dynamics matrices, A, B as a tuple
        """
        A, B = self.quad.GetLinearizedDynamics(x, u)
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0],))
        [Ad, Bd, _, _, _] = cont2discrete((A, B, C, D), self.dt)
        return Ad, Bd
         

    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> float:
        """
        :param xk: state
        :param uk: input
        :return: l(xk, uk), the running cost incurred by xk, uk
        """

        lqr_cost = 0.5 * ((xk - self.xf).T @ self.Q @ (xk - self.xf) +
                          (uk - self.quad.u_f).T @ self.R @ (uk - self.uf))

        return lqr_cost
    

    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        """
        grad = np.zeros((10,))

        #TODO: Compute the gradient
        grad[:8] = (xk - self.xf).T @ self.Q
        grad[8:] = (uk - self.uf).T @ self.R

        return grad
    

    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        """
        H = np.zeros((10, 10))

        # TODO: Compute the hessian
        H[:8,:8] = self.Q
        H[8:,8:] = self.R

        return H

    def terminal_cost(self, xf: np.ndarray) -> float:
        """
        :param xf: state
        :return: Lf(xf), the running cost incurred by xf
        """
        return 0.5 * (xf - self.xf).T @ self.Qf @ (xf - self.xf)
    

    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂Lf/∂xf
        """

        grad = np.zeros((8))

        # TODO: Compute the gradient
        grad = (xf - self.xf).T @ self.Qf
        return grad
    
        
    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂²Lf/∂xf²
        """ 
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
          uk = uu[k] + KK[k] @ delta_xk + self.alpha * dd[k]
          utraj[k] = np.clip(uk, self.quad.input_min, self.quad.input_max)
          #A, B = self.get_linearized_discrete_dynamics(xtraj[k], utraj[k])
          xtraj[k+1] = self.dynamics(xtraj[k], utraj[k])
          

        # TODO: compute forward pass
        return xtraj, utraj
    

    def backward_pass(self, xx, uu):
        """
        :param xx: state trajectory guess, should be length N
        :param uu: input trajectory guess, should be length N-1
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        dd = [np.zeros((2,))] * (self.N - 1)
        KK = [np.zeros((2, 8))] * (self.N - 1)
        H_last = self.hess_terminal_cost(xx[-1])
        g_last = self.grad_terminal_cost(xx[-1])

        for k in range(self.N-2,-1,-1):
            Ak, Bk = self.get_linearized_discrete_dynamics(xx[k], uu[k])
            Hb = hessian(self.sdf.barrier_func, torch.Tensor(xx[k]))
            gb = jacobian(self.sdf.barrier_func, torch.Tensor(xx[k]))

            Hl = self.hess_running_cost(xx[k], uu[k])
            gl = self.grad_running_cost(xx[k], uu[k])
            Hl[:8,:8] += Hb.numpy()
            gl[:8] += gb.numpy()

            Qx = gl[:8] + Ak.T @ g_last
            Qu = gl[8:] + Bk.T @ g_last
            Qxx = Hl[:8,:8] + Ak.T @ H_last @ Ak
            Quu = Hl[8:,8:] + Bk.T @ H_last @ Bk
            Qux = Hl[8:,:8] + Bk.T @ H_last @ Ak
            
            KK[k] = -np.linalg.solve(Quu, Qux)
            dd[k] = -np.linalg.solve(Quu, Qu)
            H_last = Qxx - KK[k].T @ Quu @ KK[k]
            g_last = Qx  - KK[k].T @ Quu @ dd[k]

        # TODO: compute backward pass

        return dd, KK
    

    def visualize_trajectory(self, xx, i):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot()

        self.sdf.plot_obs(ax)
        xx = np.array([xk[:2] for xk in xx])

        ax.plot(xx[:,0], xx[:,1], '--', label='actual trajectory')
        ax.axis('equal')
        ax.set_title("iter: " + str(i))
        plt.show()


    def calculate_optimal_trajectory(self, uu_guess, dt):

        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        self.dt = dt
        self.N = len(uu_guess) + 1

        # Get an initial, dynamically consistent guess for xx by simulating the quadrotor
        xx = [self.x0]
        for k in range(self.N-1):
            xx.append(self.dynamics(xx[k], uu_guess[k]))

        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        print(Jnext)
        uu = uu_guess
        KK = None

        i = 0
        print(f'cost: {Jnext}')
        while np.abs(Jprev - Jnext) > 1e-3 and i < 100:
            if self.visualize:
                self.visualize_trajectory(xx, i)

            dd, KK = self.backward_pass(xx, uu)
            xx, uu = self.forward_pass(xx, uu, dd, KK)
            
            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            print(f'cost: {Jnext}')
            i += 1
        print(f'Converged to cost {Jnext}')
        return xx, uu, KK
