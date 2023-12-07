import numpy as np
from pydrake.all import MathematicalProgram, Solve
import pydrake.symbolic as sym

def optimize_quadrotor_trajectory(quadrotor_pendulum, T, N, initial_trajectory, obstacles):
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
    dt = T / N  # Time step
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
    for i in range(N):
        x_next = x_vars[i] + dt * quadrotor_pendulum.evaluate_f(u_vars[i], x_vars[i])
        prog.AddConstraint(x_vars[i + 1] == x_next)

    # Add obstacle avoidance constraints
    for i in range(N + 1):
        for obs in obstacles:
            x_min, y_min, x_max, y_max = obs

            # Get the positions of the tips from the current state
            xr, yr, xl, yl, xm, ym = quadrotor_pendulum.get_ends(x_vars[i])

            # Keep tips within window
            if x_min <= xr <= x_max or x_min <= xm <= x_max or x_min <= xl <= x_max:
                print("In window")
                prog.AddConstraint((yr < y_min and yr < y_max) or (yr > y_min and yr > y_max))
                prog.AddConstraint((yl < y_min and yl < y_max) or (yl > y_min and yl > y_max))
                prog.AddConstraint((ym < y_min and ym < y_max) or (ym > y_min and ym > y_max))

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

# Usage example
# x0 = np.array([initial_state])  # Replace with your initial state
# xf = np.array([final_state])    # Replace with your final state
# T = 1.0  # Total time
# N = 50   # Number of time steps
# initial_trajectory = (initial_states, initial_controls)  # Replace with your initial trajectory
# optimized_states, optimized_controls = optimize_quadrotor_trajectory(quadrotor_pendulum, x0, xf, T, N, initial_trajectory)
