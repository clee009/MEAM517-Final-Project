from xml.etree.ElementPath import xpath_tokenizer_re
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math

class TrajectoryOptimizer:
    def __init__(self, quadrotor, obstacles, initial_trajectory, N, dt, goal, max_iter, tol):
        self.quadrotor = quadrotor
        self.obstacles = obstacles
        self.initial_trajectory = initial_trajectory
        self.N = N
        self.goal = goal
        self.max_iter = max_iter
        self.tol = tol
        self.state_shape = (N, 8)
        self.input_shape = (N, 2)
        self.dt = dt

    def optimize_trajectory(self):
        flat_trajectory = self.flatten_trajectory(self.initial_trajectory)
        constraints = [
            {'type': 'eq', 'fun': self.dynamics_constraint},
            {'type': 'eq', 'fun': self.initial_state_constraint, 'args': (self.initial_trajectory['state'][0],)}
        ]
        result = minimize(self.cost_function, flat_trajectory, method='SLSQP', args=(self.state_shape, self.input_shape, self.goal), 
                          constraints=constraints, options={'maxiter': self.max_iter, 'ftol': self.tol})
        
        optimized_trajectory = self.reconstruct_trajectory(result.x, self.state_shape, self.input_shape)
        self.visualize_trajectory(self.initial_trajectory, optimized_trajectory)
        return optimized_trajectory

    def cost_function(self, flat_trajectory, state_shape, input_shape, goal):
        trajectory = self.reconstruct_trajectory(flat_trajectory, state_shape, input_shape)
        states = trajectory['state']
        inputs = trajectory['input']

        energy_cost = 0
        distance_cost = 0
        goal_cost = 0

        for i in range(1, states.shape[0]):
            u = inputs[i]
            energy_cost += np.sum(u**2)  # Energy cost: sum of squared inputs

            x = states[i]
            distance_cost += np.linalg.norm(x[i][:2] - x[i-1][:2])  # Distance to goal

        goal_cost = np.linalg.norm(x[-1] - goal)

        total_cost = energy_cost +distance_cost + goal_cost # Combine costs

        print("cost =", total_cost)
        print("states =", states[self.N // 2])
        print("inputs =", inputs[self.N // 2])
        print()
        return total_cost
        
    def dynamics_constraint(self, flat_trajectory):
        trajectory = self.reconstruct_trajectory(flat_trajectory, self.state_shape, self.input_shape)
        states = trajectory['state']
        inputs = trajectory['input']

        # Obtain linearized dynamics matrices
        A, B = self.quadrotor.discrete_time_linearized_dynamics(self.dt, self.quadrotor.x_f, self.quadrotor.u_f)
        
        constraint_violations = []
        for i in range(states.shape[0] - 1):
            current_state = states[i]
            next_state = states[i + 1]
            input = inputs[i]

            # Predict the next state using the linearized dynamics
            predicted_next_state = A @ (current_state - self.quadrotor.x_f) + B @ input + self.quadrotor.x_f

            # Calculate the difference between the predicted and actual next state
            state_diff = np.linalg.norm(predicted_next_state - next_state)
            constraint_violations.append(state_diff)

        return np.array(constraint_violations)
    
    def initial_state_constraint(self, flat_trajectory, initial_state):
        trajectory = self.reconstruct_trajectory(flat_trajectory, self.state_shape, self.input_shape)
        first_state = trajectory['state'][0]
        
        # Calculate the difference between the first state and the initial state
        state_diff = np.linalg.norm(first_state - initial_state)
        return state_diff

    def flatten_trajectory(self, trajectory):
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

    def reconstruct_trajectory(self, flat_trajectory, state_shape, input_shape):
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

    def visualize_trajectory(self, initial_trajectory, optimized_trajectory):
        # Visualization of initial and optimized trajectories
        plt.figure()
        plt.plot(initial_trajectory['state'][:, 0], initial_trajectory['state'][:, 1], label='Initial Trajectory')
        plt.plot(optimized_trajectory['state'][:, 0], optimized_trajectory['state'][:, 1], label='Optimized Trajectory')
        plt.legend()
        plt.show()

# Usage
# optimizer = TrajectoryOptimizer(quadrotor, obstacles, initial_trajectory, N, goal, max_iter, tol)
# optimized_trajectory = optimizer.optimize_trajectory()
