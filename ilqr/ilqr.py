import numpy as np
from obstacles import Obstacles
from quadrotor_with_pendulum import QuadrotorPendulum


class iLQR:
    def __init__(self, quad: QuadrotorPendulum, obs: Obstacles):
        self.quad = quad
        self.obs = obs