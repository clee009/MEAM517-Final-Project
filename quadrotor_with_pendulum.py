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

# Define a system to calculate the continuous dynamics
# of the quadrotor pendulum.
# 
# This class takes as input the physical description
# of the system, in terms of the center of mass of 
# the drone (mb with body lenght lb) and the first
# link (m1 centered at l1).
class QuadrotorPendulum(VectorSystem):
    def __init__(self, Q, R, Qf, x_f, mb = 1., 
                        lb = 0.2, m1 = 2., l1 = 0.2,
                        g = 10., input_max = 30.):
        VectorSystem.__init__(self,
            2,                           # Two input (thrust of each rotor).
            8)                           # Eight outputs (xb, yb, thetab, theta1) and its derivatives
        # self._DeclareContinuousState(8)  # Eight states (xb, yb, thetab, theta1) and its derivatives.

        self.mb = float(mb)
        self.lb = float(lb)
        self.m1 = float(m1)
        self.l1 = float(l1)
        self.g = float(g)
        self.input_max = float(input_max)
        self.input_min = 0.0

        # Go ahead and calculate rotational inertias.
        # Treat the drone as a rectangle.
        self.Ib = 1. / 3. * self.mb * self.lb ** 2
        # Treat the first link as a line.
        self.I1 = 1. / 3. * self.m1 * self.l1 ** 2

        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Nominal state and input for linearization
        self.x_f = x_f
        self.u_f = np.array([(self.mb + self.m1)*self.g/2, (self.mb + self.m1)*self.g/2])

    # This method returns (M, C, tauG, B)
    # according to the dynamics of this system.
    def GetManipulatorDynamics(self, q, qd):
        M = np.array(
            [[self.mb + self.m1, 0., 0., self.m1*self.l1*math.cos(q[3])],
                [0., self.mb + self.m1, 0., self.m1*self.l1*math.sin(q[3])],
                [0., 0., self.Ib, 0.],
                [self.m1*self.l1*math.cos(q[3]), self.m1*self.l1*math.sin(q[3]), 0., self.I1 + self.m1*self.l1**2]])

        C = np.array(
            [[0., 0., 0., -self.m1*self.l1*math.sin(q[3])*qd[3]],
                [0., 0., 0., self.m1*self.l1*math.cos(q[3])*qd[3]],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]])

        tauG = np.array(
            [[0.],
                [-(self.m1+self.mb)*self.g],
                [0.],
                [-self.m1*self.l1*self.g*math.sin(q[3])]])

        B = np.array(
            [[-math.sin(q[2]), -math.sin(q[2])],
                [math.cos(q[2]), math.cos(q[2])],
                [-self.lb, self.lb],
                [0., 0.]])

        return (M, C, tauG, B)
  
    def get_ends(self, x):
        """
        Returns the positions of the ends of the quadrotor (tips of body and pendulum).

        OUTPUTS:
        end_pos - 6 element numpy array, [xr, yr, xl, yl, xm, ym]
        """

        xb = x[0]
        yb = x[1]
        thb = x[2]
        th1 = x[3]

        # Right tip of body/wing
        xr = xb + self.lb / 2 * cos(thb)
        yr = yb + self.lb / 2 * sin(thb)

        # Left tip of body/wing
        xl = xb - self.lb / 2 * cos(thb)
        yl = yb - self.lb / 2 * sin(thb)

        # Tip of pendulum
        xm = xb + self.l1 * sin(th1)
        ym = yb - self.l1 * cos(th1)

        end_pos = np.array([[xr, yr], [xl, yl], [xm, ym]])

        return end_pos

    # This helper uses the manipulator dynamics to evaluate
    # \dot{x} = f(x, u). It's just a thin wrapper around
    # the manipulator dynamics. If throw_when_limits_exceeded
    # is true, this function will throw a ValueError when
    # the input limits are violated. Otherwise, it'll clamp
    # u to the input range.
    def evaluate_f(self, u, x, throw_when_limits_exceeded=False):
        # Bound inputs
        if throw_when_limits_exceeded and abs(u[0]) > self.input_max:
            raise ValueError("You commanded an out-of-range input of u=%f" % (u[0]))
        else:
            u[0] = max(-self.input_max, min(self.input_max, u[0]))

        if throw_when_limits_exceeded and abs(u[1]) > self.input_max:
            raise ValueError("You commanded an out-of-range input of u=%f" % (u[1]))
        else:
            u[1] = max(-self.input_max, min(self.input_max, u[1]))

        # Use the manipulator equation to get qdd.
        q = x[0:4]
        qd = x[4:8]
        (M, C, tauG, B) = self.GetManipulatorDynamics(q, qd)

        # Awkward slice required on tauG to get shapes to agree --
        # numpy likes to collapse the other dot products in this expression
        # to vectors.
        qdd = np.dot(np.linalg.inv(M), (tauG[:, 0] + np.dot(B, u) - np.dot(C, qd)))

        # print('evaluate_f output =', np.hstack([qd, qdd]))

        return np.hstack([qd, qdd])

    #   # This method calculates the time derivative of the state,
    #   # which allows the system to be simulated forward in time.
    #   def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
    #     q = x[0:4]
    #     qd = x[4:8]
    #     xdot[:] = self.evaluate_f(u, x, throw_when_limits_exceeded=False)

    #   # This method calculates the output of the system
    #   # (i.e. those things that are visible downstream of
    #   # this system) from the state. In this case, it
    #   # copies out the full state.
    #   def _DoCalcVectorOutput(self, context, u, x, y):
    #     y[:] = x

    #   # The Drake simulation backend is very careful to avoid
    #   # algebraic loops when systems are connected in feedback.
    #   # This system does not feed its inputs directly to its
    #   # outputs (the output is only a function of the state),
    #   # so we can safely tell the simulator that we don't have
    #   # any direct feedthrough.
    #   def _DoHasDirectFeedthrough(self, input_port, output_port):
    #     if input_port == 0 and output_port == 0:
    #       return False
    #     else:
    #       # For other combinations of i/o, we will return
    #       # "None", i.e. "I don't know."
    #       return None

    # The method return matrices (A) and (B) that encode the
    # linearized dynamics of this system around the fixed point
    # u_f, x_f.
    def GetLinearizedDynamics(self, x_f, u_f):
        q = x_f[0:4]
        qd = x_f[4:8]

        A = np.zeros((8, 8))
        B = np.zeros((8, 2))
        C = np.zeros((8, 1))

        alpha = self.m1*self.l1/(self.m1+self.mb)
        I = self.I1 + self.m1*self.mb*self.l1**2/(self.m1+self.mb)

        B[4,0] = -1/(self.m1+self.mb)*math.sin(q[2]) + alpha**2/I*math.cos(q[3])*math.sin(q[3]-q[2])
        B[4,1] = B[4,0]

        B[5,0] = (self.I1+self.m1*self.l1**2)/(self.m1+self.mb)/I*math.cos(q[2]) - alpha**2/I*math.cos(q[3])*math.cos(q[3]-q[2])
        B[5,1] = B[5,0]

        B[6,0] = -self.lb/self.Ib
        B[6,1] = self.lb/self.Ib

        B[7,0] = -alpha/I*math.sin(q[3]-q[2])
        B[7,1] = -alpha/I*math.sin(q[3]-q[2])

        A[:4,4:] = np.diag(np.ones(4))

        A[4,2] = (-1/(self.m1+self.mb)*math.cos(q[2])-alpha**2/I*math.cos(q[3])*math.cos(q[3]-q[2]))*(u_f[0]+u_f[1])
        A[5,2] = (-(self.I1+self.m1*self.l1**2)/(self.m1+self.mb)/I*math.sin(q[2])-alpha**2/I*math.cos(q[3])*math.sin(q[3]-q[2]))*(u_f[0]+u_f[1])

        A[4,3] = alpha*math.cos(q[3])*qd[3]**2+alpha**2/I*math.cos(2.*q[3]-q[2])*(u_f[0]+u_f[1])
        A[5,3] = alpha*math.sin(q[3])*qd[3]**2+alpha**2/I*math.sin(2.*q[3]-q[2])*(u_f[0]+u_f[1])
        A[4,7] = 2.*alpha*math.sin(q[3])*qd[3]
        A[5,7] = -2.*alpha*math.cos(q[3])*qd[3]

        A[7,2] = alpha/I*math.cos(q[3]-q[2])*(u_f[0]+u_f[1])
        A[7,3] = -alpha/I*math.cos(q[3]-q[2])*(u_f[0]+u_f[1])

        C[4] = alpha*math.sin(q[3])*qd[3]**2+(-1/(self.m1+self.mb)*math.sin(q[2]) + alpha**2/I*math.cos(q[3])*math.sin(q[3]-q[2]))*(u_f[0]+u_f[1])
        C[5] = -alpha*math.cos(q[3])*qd[3]**2-self.g+((self.I1+self.m1*self.l1**2)/(self.m1+self.mb)/I*math.cos(q[2])-alpha**2/I*math.cos(q[3])*math.cos(q[3]-q[2]))*(u_f[0]+u_f[1])
        C[6] = (-u_f[0]+u_f[1])/self.Ib
        C[7] = -alpha/I*math.sin(q[3]-q[2])*(u_f[0]+u_f[1])

        return (A, B)
  
    def discrete_time_linearized_dynamics(self, dt, x_f, u_f):
        # dt is the time step for discretization
        # Get the continuous-time linearized dynamics
        A_c, B_c = self.GetLinearizedDynamics(x_f, u_f)

        # Discretize the linearized dynamics
        I = np.identity(len(x_f))  # Identity matrix
        A_d = I + A_c * dt
        B_d = B_c * dt

        return A_d, B_d
  
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

        A, B = self.discrete_time_linearized_dynamics(dt, self.x_f, self.u_f)

        for k, (xk, xk_p1) in enumerate(zip(x, x[1:])):
            prog.AddLinearEqualityConstraint(xk_p1 - A @ xk - B @ u[k], np.zeros_like(xk))

    def add_cost(self, prog, x, u):
        # TODO: add cost.
        cost = x[-1].T @ self.Qf @ x[-1]
        i = 1

        for xk, uk in zip(x, u):
            xf = x[i]
            xe = xk - xf
            ue = uk - self.u_f

            cost += xe.T @ self.Q @ xe
            cost += ue.T @ self.R @ ue
            
            i += 1

        prog.AddQuadraticCost(cost)

    def compute_mpc_feedback(self, x_current, use_clf=False):
        '''
        This function computes the MPC controller input u
        '''

        # Parameters for the QP
        N = 10
        dt = 0.1

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
        self.add_input_saturation_constraint(prog, u)
        self.add_dynamics_constraint(prog, x, u, dt)
        self.add_cost(prog, x, u)

        # Placeholder constraint and cost to satisfy QP requirements
        # TODO: Delete after completing this function

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)


        u_mpc = result.GetSolution(u[0])
        # TODO: retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        # You should make use of result.GetSolution(decision_var) where decision_var
        # is the variable you want

        return u_mpc + self.u_f

    def compute_lqr_feedback(self, x_current):
        '''
        Infinite horizon LQR controller
        '''
        u_f = self.u_d()
        x_f = self.x_d()

        A, B = self.GetLinearizedDynamics(u_f, x_f)
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -inv(self.R) @ B.T @ S
        u = u_f + K @ (x_current - x_f)
        return u
