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

class TrajectoryOptimizer:
    def __init__(self, quadrotor):
        self.quadrotor = quadrotor
        self.x_f = quadrotor.x_f
        self.u_f = quadrotor.u_f
        self.Qf = quadrotor.Qf
        self.Q = quadrotor.Q
        self.R = quadrotor.R
        self.input_min = quadrotor.input_min
        self.input_max = quadrotor.input_max
        

    def add_initial_state_constraint(self, prog, x, x_curr):
        # TODO: impose initial state constraint.
        # Use AddBoundingBoxConstraint
        prog.AddBoundingBoxConstraint(x_curr, x_curr, x[0])

    def add_input_saturation_constraint(self, prog, u):
        # TODO: impose input limit constraint.
        # Use AddBoundingBoxConstraint
        # The limits are available through self.umin and self.umax
        for uk in u:
            ones = np.ones_like(uk)
            prog.AddBoundingBoxConstraint(self.input_min * ones - self.u_f, self.input_max * ones - self.u_f, uk)

    def add_dynamics_constraint(self, prog, x, u, dt):
        # TODO: impose dynamics constraint.
        # Use AddLinearEqualityConstraint(expr, value)

        A, B = self.quadrotor.discrete_time_linearized_dynamics(dt, self.x_f, self.u_f)

        for k, (xk, xk_p1) in enumerate(zip(x, x[1:])):
            # prog.AddLinearEqualityConstraint((xk_p1 - self.x_f) - A @ (xk - self.x_f) - B @ u[k], np.zeros_like(xk))
            prog.AddLinearEqualityConstraint(xk_p1 - A @ xk - B @ u[k], np.zeros_like(xk))

    def add_cost(self, prog, x, u):
        # TODO: add cost.
        xe = x[-1] - self.x_f
        cost = xe.T @ self.Qf @ xe

        for xk, uk in zip(x, u):
            xe = xk - self.x_f

            cost += xe.T @ self.Q @ xe
            cost += uk.T @ self.R @ uk

        prog.AddQuadraticCost(cost)

    def add_obstacle_constraints(self, prog, x, obstacles):
        for xk in x:
            # Get the end positions (e.g., tips) of the quadrotor for the current state
            end_positions = self.quadrotor.get_ends(xk)
            
            # Add a constraint for each obstacle
            for box in obstacles.boxes:
                # Use symbolic expressions for the feasibility check
                feasible_expr = obstacles.is_feasible_continuous(end_positions)
                
                # Add a constraint that feasible_expr must be greater than zero
                prog.AddConstraint(feasible_expr >= 0)
    
    def compute_feedback(self, x_current, t_step, N, obstacles, initial_traj=None):
        '''
        This function computes the controller input u
        '''

        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, 8), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(8, "x_" + str(i))
        u = np.zeros((N-1, 2), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        if initial_traj is not None:
            for i in range(N):
                if i < len(initial_traj['state']):
                    prog.SetInitialGuess(x[i], initial_traj['state'][i])
                if i < len(initial_traj['input']) and i < N - 1:
                    prog.SetInitialGuess(u[i], initial_traj['input'][i])

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)
        self.add_input_saturation_constraint(prog, u)
        self.add_dynamics_constraint(prog, x, u, t_step)
        self.add_cost(prog, x, u)
        self.add_obstacle_constraints(prog, x, obstacles)

        # Placeholder constraint and cost to satisfy QP requirements
        # TODO: Delete after completing this function

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)
        print(result)


        u_mpc = result.GetSolution(u[0])
        # TODO: retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        # You should make use of result.GetSolution(decision_var) where decision_var
        # is the variable you want

        return u_mpc + self.u_f
    

    def optimize_trajectory(self, x0, N, t_step, obstacles, initial_traj=None):
        prog = MathematicalProgram()
        x = np.zeros((N, 8), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(8, "x_" + str(i))
        u = np.zeros((N-1, 2), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        if initial_traj is not None:
            for i in range(N):
                if i < len(initial_traj['state']):
                    prog.SetInitialGuess(x[i], initial_traj['state'][i])
                if i < len(initial_traj['input']) and i < N - 1:
                    prog.SetInitialGuess(u[i], initial_traj['input'][i])

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x0)
        self.add_input_saturation_constraint(prog, u)
        self.add_dynamics_constraint(prog, x, u, t_step)
        self.add_cost(prog, x, u)
        self.add_obstacle_constraints(prog, x, obstacles)

        # Placeholder constraint and cost to satisfy QP requirements
        # TODO: Delete after completing this function

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)
        x = result.GetSolution(x)
        return x
        # TODO: retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        # You should make use of result.GetSolution(decision_var) where decision_var
        # is the variable you want
    
    def simulate_quadrotor(self, x0, tf, xf, N, t_step, obstacles, initial_traj=None):
        # Simulates a stabilized maneuver on the 2D quadrotor
        # system, with an initial value of x0
        t0 = 0.0
        # n_points = 1000

        # dt for quadrotor.evaluate_f
        dt = 1e-2

        x = [x0]
        u = [np.zeros((2,))]
        t = [t0]

        while np.linalg.norm(np.array(x[-1][0:2]) - xf[0:2]) > 1e-3 and t[-1] < tf:
            current_time = t[-1]
            current_x = x[-1]
            current_u_command = np.zeros(2)

            current_u_command = self.compute_feedback(current_x, t_step, N, obstacles, initial_traj)

            current_u_real = np.clip(current_u_command, self.quadrotor.input_min, self.quadrotor.input_max)
            # Autonomous ODE for constant inputs to work with solve_ivp
            def f(t, x):
                return self.quadrotor.evaluate_f(current_u_real, current_x)
            
            # Integrate one step
            sol = solve_ivp(f, (0, dt), current_x, first_step=dt)

            # Record time, state, and inputs
            t.append(t[-1] + dt)
            x.append(sol.y[:, -1])
            u.append(current_u_command)

        x = np.array(x)
        u = np.array(u)
        t = np.array(t)
        return x, u, t