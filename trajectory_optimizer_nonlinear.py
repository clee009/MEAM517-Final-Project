import numpy as np
from pydrake.all import MathematicalProgram, Solve
import pydrake.symbolic as sym
from obstacles import Obstacles

def optimize_quadrotor_trajectory(quadrotor_pendulum, N, dt, initial_trajectory, obstacles):
    """
    Optimizes the trajectory for a quadrotor with a pendulum using an initial guess.

    Parameters:
    quadrotor_pendulum (QuadrotorPendulum): The quadrotor pendulum system.
    x0 (np.array): Initial state of the system.
    xf (np.array): Final desired state of the system.
    T (float): Total time for the trajectory.
    N (int): Number of discrete time steps.
    initial_trajectory (tuple): Tuple of two arrays (initial_states, initial_controls),
                               representing the initial guess for states and controls.

    Returns:
    np.array, np.array: Optimized states and controls.
    """
    initial_states, initial_controls = initial_trajectory

    x0 = initial_states[0]
    xf = initial_states[-1]

    # Create an instance of MathematicalProgram
    prog = MathematicalProgram()

    # Define state and control variables
    x_vars = np.array([prog.NewContinuousVariables(8, f"x_{i}") for i in range(N + 1)])
    u_vars = np.array([prog.NewContinuousVariables(2, f"u_{i}") for i in range(N)])

    # Set initial and final state constraints
    prog.AddBoundingBoxConstraint(x0, x0, x_vars[0])
    prog.AddBoundingBoxConstraint(xf, xf, x_vars[-1])

    # Add dynamic constraints
    A, B = quadrotor_pendulum.discrete_time_linearized_dynamics(dt)
    for i, (xi, xi_p1) in enumerate(zip(x_vars, x_vars[1:])):
      prog.AddLinearEqualityConstraint(xi_p1 - A @ xi - B @ u_vars[i], np.zeros_like(xi))

    for i in range(N):
        # Add obstacle avoidance constraints
        tip_pos = quadrotor_pendulum.get_ends(x_vars[i])
        feasibility = obstacles.is_feasible_continuous(tip_pos)
        prog.AddConstraint(feasibility >= 0)
    
    # Add boundary constraints
    x_min, y_min, x_max, y_max = obstacles.get_world()[0]
    for i in range(N + 1):
        # Extract the position (x, y) from the state
        x_pos = x_vars[i][0]
        y_pos = x_vars[i][1]

        # Add boundary constraints
        prog.AddConstraint(x_pos >= x_min)
        prog.AddConstraint(x_pos <= x_max)
        prog.AddConstraint(y_pos >= y_min)
        prog.AddConstraint(y_pos <= y_max)

    # Define and add the cost function (Modify as per your specific cost function)
    for u in u_vars:
        prog.AddQuadraticCost(np.dot(u, u))

    # Set the initial guess for the optimization
    for i in range(N + 1):
        prog.SetInitialGuess(x_vars[i], initial_states[i])
    for i in range(N):
        prog.SetInitialGuess(u_vars[i], initial_controls[i])

    # Solve the optimization problem
    result = Solve(prog)
    if not result.is_success():
        raise ValueError("Optimization failed")

    # Extract the optimized trajectory
    optimized_states = np.array([result.GetSolution(x) for x in x_vars])
    optimized_controls = np.array([result.GetSolution(u) for u in u_vars])

    return optimized_states, optimized_controls