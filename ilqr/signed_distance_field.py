import pydrake.all as drake
import numpy as np
from world import Obstacles


class SignedDistanceField:
    def __init__(self, obs: Obstacles):
        self.obs = obs
        self.a = obs.get_world()