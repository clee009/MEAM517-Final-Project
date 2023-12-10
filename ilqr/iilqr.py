import numpy as np
from scipy.signal import cont2discrete
from typing import List, Tuple
import torch
from torch.autograd.functional import hessian, jacobian
from scipy.integrate import solve_ivp

from lqrrt import PathPlannerLQRRT

import configs
from ilqr import iLQR


class iiLQR(iLQR):
    def __init__(self, file: str, rrt: PathPlannerLQRRT):
        super().__init__(file, rrt)
        

    def calculate_optimal_trajectory(self, x0, xx_guess, uu_guess, dt):
        xx_traj = [np.copy(x0)]
        uu_traj = []

        uu = []
        for k in range(1, len(xx_guess)):
            x0 = xx_traj[-1]
            xf = xx_guess[k+1]
            uk = uu_guess[k]

            uu.append(uk)
            infeasible_k, xx, uu_temp = super().calculate_optimal_trajectory(x0, xf, uu, dt)
            if infeasible_k != -1: #success!
                uu = uu_temp
                continue

            
            assert (xx_traj[-1] == xx[0]).all()
            uu_traj += uu
            xx_traj += xx[1:]

            uu = []

        return xx_traj, uu_traj
