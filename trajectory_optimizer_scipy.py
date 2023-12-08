from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math

# Define the cost function
def cost_function(flat_trajectory, state_shape, input_shape, goal):

    trajectory = reconstruct_trajectory(flat_trajectory, state_shape, input_shape)

    # Calculate the cost based on input energy and distance
    states = trajectory['state']
    inputs = trajectory['input']

    energy_cost = 0
    goal_distance_cost = 0
    trajectory_length_cost = 0

    for i in range(1, states.shape[0]):
        # Energy cost calculation
        u = inputs[i]
        energy_cost += np.sum(u**2)

        # Distance calculation
        # prev_state = states[i - 1]
        # current_state = states[i]
        
        # distance = np.linalg.norm(current_state[:2] - prev_state[:2])
        # trajectory_length_cost += distance

    final_state = states[-1]  # Assuming the final state contains position info
    goal_distance_cost = np.linalg.norm(final_state[:2] - goal[:2])  # Assuming 2D position is the first two elements

    # Combine the costs
    total_cost = energy_cost + goal_distance_cost + trajectory_length_cost

    print('cost =', total_cost)
    
    return total_cost

# Define the constraints for obstacle avoidance
def strict_obstacle_constraint(flat_trajectory, quadrotor, obstacles, state_shape, input_shape):

    trajectory = reconstruct_trajectory(flat_trajectory, state_shape, input_shape)

    # Check if the trajectory intersects with any obstacle
    for state in trajectory['state']:
        tip_pos = quadrotor.get_ends(state)
        is_feasible = obstacles.is_feasible(tip_pos)
        if not is_feasible:
            return -1
        
    return 1

def flatten_trajectory(trajectory):
    """
    Flattens the trajectory dictionary into a 1D array.

    Parameters:
    trajectory (dict): Trajectory dictionary with keys 'state' and 'input'.

    Returns:
    numpy.ndarray: Flattened trajectory as a 1D array.
    """
    x_flat = trajectory['state'].flatten()
    u_flat = trajectory['input'].flatten()
    return np.concatenate([x_flat, u_flat])

def reconstruct_trajectory(flat_trajectory, state_shape, input_shape):
    """
    Reconstructs the trajectory dictionary from a 1D array.

    Parameters:
    flat_trajectory (numpy.ndarray): Flattened trajectory as a 1D array.
    state_shape (tuple): Shape of the state array.
    input_shape (tuple): Shape of the input array.

    Returns:
    dict: Trajectory dictionary with keys 'state' and 'input'.
    """
    total_state_elements = np.prod(state_shape)
    x_flat = flat_trajectory[:total_state_elements]
    u_flat = flat_trajectory[total_state_elements:]
    
    x = x_flat.reshape(state_shape)
    u = u_flat.reshape(input_shape)

    return {'state': x, 'input': u}


def trajectory_optimizer(quadrotor, obstacles, initial_trajectory, N, max_iter, goal):
    """
    """

    trajectory = flatten_trajectory(initial_trajectory)
    
    state_shape = (N, 8)
    input_shape = (N, 2)

    constraints = [{'type': 'ineq', 'fun': strict_obstacle_constraint, 'args': (quadrotor, obstacles, state_shape, input_shape)}]

    # Optimization problem setup
    result = minimize(cost_function, trajectory, args = (state_shape, input_shape, goal), options={'maxiter': max_iter}, constraints=constraints)

    # Extract the optimized trajectory
    optimized_trajectory = reconstruct_trajectory(result.x, state_shape, input_shape)

    # Visualization of initial and optimized trajectories
    plt.figure()
    plt.plot(initial_trajectory['state'][:, 0], initial_trajectory['state'][:, 1], label='Initial Trajectory')
    plt.plot(optimized_trajectory['state'][:, 0], optimized_trajectory['state'][:, 1], label='Optimized Trajectory')
    plt.legend()
    plt.show()

    return optimized_trajectory