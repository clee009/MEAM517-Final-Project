import numpy as np
import numpy.linalg as npl
import lqrrt
import configs
from quadrotor_with_pendulum import QuadrotorPendulum

class KinodtnamicRRT:
    def __init__(self, file: str, quad: QuadrotorPendulum):
        for key, value in configs.load_yaml(file).items():
            setattr(self, key, value)

        self.R = np.array(self.R)
        self.iR = np.linalg.inv(self.R)

        self.Q = np.array(self.Q)
        self.Qf = np.array(self.Qf)
        self.quad = quad


    def lqr(self, x, u): #lqrrt input wrapping
        A, B = self.quad.GetLinearizedDynamics(u, x)

        S = np.identity(len(self.Q)) #todo : implement ricatti
        K = -self.iR @ B.T @ S

        return S, K