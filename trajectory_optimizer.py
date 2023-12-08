import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import VectorSystem, MonomialBasis, OddDegreeMonomialBasis, Variables

class trajectory_optimizer:
    def __init__(self, quadrotor, ):
        pass

    def add_initial_state_constraint(self, prog, x, x_curr):
        # TODO: impose initial state constraint.
        # Use AddBoundingBoxConstraint
        prog.AddBoundingBoxConstraint(x_curr, x_curr, x[0])

    def add_input_saturation_constraint(self, prog, x, u, N):
        # TODO: impose input limit constraint.
        # Use AddBoundingBoxConstraint
        # The limits are available through self.umin and self.umax
        for uk in u:
            ones = np.ones_like(uk)
            prog.AddBoundingBoxConstraint(self.input_min * ones - self.u_d(), self.input_max * ones - self.u_d(), uk)

    def add_dynamics_constraint(self, prog, x, u, N, T):
        # TODO: impose dynamics constraint.
        # Use AddLinearEqualityConstraint(expr, value)
        A, B = self.discrete_time_linearized_dynamics(T)
        for k, (xk, xk_p1) in enumerate(zip(x, x[1:])):
            prog.AddLinearEqualityConstraint(xk_p1 - A @ xk - B @ u[k], np.zeros_like(xk))

    def add_cost(self, prog, x, u, N):
        # TODO: add cost.
        cost = x[-1].T @ self.Qf @ x[-1]
        for xk, uk in zip(x, u):
            cost += xk.T @ self.Q @ xk
            cost += uk.T @ self.R @ uk

        prog.AddQuadraticCost(cost)

    def compute_feedback(self, x_current):
        '''
        This function computes the MPC controller input u
        '''

        # Parameters for the QP
        N = 10
        T = 0.1

        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, 8), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(8, "x_" + str(i))
        u = np.zeros((N-1, 2), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)
        self.add_input_saturation_constraint(prog, x, u, N)
        self.add_dynamics_constraint(prog, x, u, N, T)
        self.add_cost(prog, x, u, N)

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)


        u_mpc = result.GetSolution(u[0])
        # TODO: retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        # You should make use of result.GetSolution(decision_var) where decision_var
        # is the variable you want

        return u_mpc + self.u_d()