import casadi as ca
import numpy as np
from world import Obstacles
import matplotlib.pyplot as plt

class SignedDistanceField(Obstacles):
    def __init__(self, file: str, gamma = 3.6):
        super().__init__(file)
        self.n = len(self.boxes)
        self.gamma = gamma
    

    def calc_sdf_single(self, state, idx):
        x, y = state[0, 0], state[0, 1]
        x_min, y_min, x_max, y_max = self.boxes[idx]

        # Using ca.if_else for conditional logic
        bottom_left = ca.sqrt((x_min - x)**2 + (y_min - y)**2)
        top_left = ca.sqrt((x_min - x)**2 + (y - y_max)**2)
        bottom_right = ca.sqrt((x - x_max)**2 + (y_min - y)**2)
        top_right = ca.sqrt((x - x_max)**2 + (y - y_max)**2)

        left_edge = x_min - x
        right_edge = x - x_max
        bottom_edge = y_min - y
        top_edge = y - y_max

        dx = ca.fmax(x - x_max, x_min - x)
        dy = ca.fmax(y - y_max, y_min - y)
        center = ca.fmax(dx, dy)

        return ca.if_else(
            x < x_min,
            ca.if_else(
                y < y_min, bottom_left,
                ca.if_else(y > y_max, top_left, left_edge)
            ),
            ca.if_else(
                x > x_max,
                ca.if_else(
                    y < y_min, bottom_right,
                    ca.if_else(y > y_max, top_right, right_edge)
                ),
                ca.if_else(
                    y < y_min, bottom_edge,
                    ca.if_else(y > y_max, top_edge, center)
                )
            )
        )

    def calc_sdf(self, x):
        min_sdf = self.calc_sdf_single(x, 0)
        for i in range(1, self.n):
            min_sdf = ca.fmin(self.calc_sdf_single(x, i), min_sdf)
        
        return min_sdf

        # return self.calc_sdf_single(x, 0)


    def barrier_func(self, x):
        epsilon = 1e-3
        sdf = self.calc_sdf(x)
        barrier_arg = sdf + epsilon
        # return ca.exp(-self.gamma * sdf)
        return -ca.log(self.gamma * barrier_arg)
    
def ellipsoidal_function(state, box, lambda_param):
    """
    """
    x, y = state[0, 0], state[0, 1]

    xmin, ymin, xmax, ymax = box

    # Define numerical values for the parameters
    c_x = (xmin + xmax) / 2   # Numerical value for the center x
    c_y = (ymin + ymax) / 2  # Numerical value for the center y
    r_x = (xmax - xmin) / 2  # Numerical value for the size in x
    r_y = (ymax - ymin) / 2  # Numerical value for the size in y

    # Define the cost function directly using numerical values
    penalty = 1 / (((x - c_x)**2 / r_x + (y - c_y)**2 / r_y)**lambda_param + 1)

    return penalty

def get_nonlinear_dynamics(q, qd, params):
        
    mb, lb, m1, l1, g = params['mb'], params['lb'], params['m1'], params['l1'], params['g']

    Ib = 1/3*mb*lb**2
    I1 = 1/3*m1*l1**2

    M = ca.MX.zeros(4, 4)
    C = ca.MX.zeros(4, 4)
    tauG = ca.MX.zeros(4, 1)
    B = ca.MX.zeros(4, 2)

    M = np.array(
        [[mb + m1, 0., 0., m1*l1*ca.cos(q[3])],
            [0., mb + m1, 0., m1*l1*ca.sin(q[3])],
            [0., 0., Ib, 0.],
            [m1*l1*ca.cos(q[3]), m1*l1*ca.sin(q[3]), 0., I1 + m1*l1**2]])

    C = np.array(
        [[0., 0., 0., -m1*l1*ca.sin(q[3])*qd[3]],
            [0., 0., 0., m1*l1*ca.cos(q[3])*qd[3]],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])

    tauG = np.array(
        [[0.],
            [-(m1+mb)*g],
            [0.],
            [-m1*l1*g*ca.sin(q[3])]])

    B = np.array(
        [[-ca.sin(q[2]), -ca.sin(q[2])],
            [ca.cos(q[2]), ca.cos(q[2])],
            [-lb, lb],
            [0., 0.]])

    return (M, C, tauG, B)

def get_linearized_dynamics(x_f, u_f, params):
    """
    """
    # Extract parameters
    mb, lb, m1, l1, g = params['mb'], params['lb'], params['m1'], params['l1'], params['g']
    
    q = x_f[0:4]
    qd = x_f[4:8]

    # Define matrices
    A = ca.MX.zeros(8, 8)
    B = ca.MX.zeros(8, 2)

    # Calculate elements of A and B
    Ib = 1/3*mb*lb**2
    I1 = 1/3*m1*l1**2

    alpha = m1*l1/(m1+mb)
    I = I1+m1*mb*l1**2/(m1+mb)

    B[4,0] = -1/(m1+mb)*ca.sin(q[2]) + alpha**2/I*ca.cos(q[3])*ca.sin(q[3]-q[2])
    B[4,1] = B[4,0]

    B[5,0] = (I1+m1*l1**2)/(m1+mb)/I*ca.cos(q[2]) - alpha**2/I*ca.cos(q[3])*ca.cos(q[3]-q[2])
    B[5,1] = B[5,0]

    B[6,0] = -lb/Ib
    B[6,1] = lb/Ib

    B[7,0] = -alpha/I*ca.sin(q[3]-q[2])
    B[7,1] = -alpha/I*ca.sin(q[3]-q[2])

    A[0:4, 4:8] = ca.MX.eye(4)

    A[4,2] = (-1/(m1+mb)*ca.cos(q[2])-alpha**2/I*ca.cos(q[3])*ca.cos(q[3]-q[2]))*(u_f[0]+u_f[1])
    A[5,2] = (-(I1+m1*l1**2)/(m1+mb)/I*ca.sin(q[2])-alpha**2/I*ca.cos(q[3])*ca.sin(q[3]-q[2]))*(u_f[0]+u_f[1])

    A[4,3] = alpha*ca.cos(q[3])*qd[3]**2+alpha**2/I*ca.cos(2*q[3]-q[2])*(u_f[0]+u_f[1])
    A[5,3] = alpha*ca.sin(q[3])*qd[3]**2+alpha**2/I*ca.sin(2*q[3]-q[2])*(u_f[0]+u_f[1])
    A[4,7] = 2*alpha*ca.sin(q[3])*qd[3]
    A[5,7] = -2*alpha*ca.cos(q[3])*qd[3]

    A[7,2] = alpha/I*ca.cos(q[3]-q[2])*(u_f[0]+u_f[1])
    A[7,3] = -alpha/I*ca.cos(q[3]-q[2])*(u_f[0]+u_f[1])

    return A, B

def discrete_time_linearized_dynamics(dt, x_f, u_f, params):
    A_c, B_c = get_linearized_dynamics(x_f.reshape(-1), u_f.reshape(-1), params)

    # Identity matrix in CasADi
    I = ca.MX.eye(8)

    # Discretize the linearized dynamics
    A_d = I + A_c * dt
    B_d = B_c * dt

    return A_d, B_d

def optimize_trajectory(quadrotor, obstacles, N, dt, initial_trajectory, alpha, lambda_param, 
                        opt_params = {
                            'max_iter': 3000,
                            'acceptable_tol': 1e-6,
                            'acceptable_constr_viol_tol': 1e-2
                        }):
    """
    """
    # Define parameters
    params = {
        "mb": quadrotor.mb,
        "lb": quadrotor.lb,
        "m1": quadrotor.m1,
        "l1": quadrotor.l1,
        "g": quadrotor.g,
        "input_max": quadrotor.input_max,
        "input_min": quadrotor.input_min
    }

    x_f = quadrotor.x_f.reshape(1, 8)
    u_f = quadrotor.u_f.reshape(1, 2)

    # Start optimization problem
    opti = ca.Opti()

    # Get obstacles
    boundary, boxes = obstacles.get_world()

    # Define decision variables for states and control inputs
    X = opti.variable(N, 8)  # State trajectory
    U = opti.variable(N-1, 2)   # Control inputs

    # Define initial guesses and initial state
    initial_state_guess = initial_trajectory['state']
    initial_control_guess = initial_trajectory['input']
    initial_state = initial_state_guess[0].reshape(1, 8)
    print('intial_state shape =', initial_state.shape)
    print('X[0] shape =', X[0, :].shape)
    final_state = initial_state_guess[-1].reshape(1, 8)

    # Set initial guesses
    opti.set_initial(X, initial_state_guess)
    opti.set_initial(U, initial_control_guess)

    # Add initial state constraint
    opti.subject_to(X[0, :] == initial_state)

    # Add final state constraint
    opti.subject_to(X[N-1, :] == final_state)

    # Add dynamics constraints
    A, B = discrete_time_linearized_dynamics(dt, x_f, u_f, params)
    print('B shape =', B)
    for k in range(N-1):
        xk = X[k, :]
        xk_next = X[k+1, :]
        uk = U[k, :]
        xe = xk - x_f
        xk_collocation = ca.mtimes(A, ca.reshape(xe, 8, 1)) + ca.mtimes(B, ca.reshape(uk, 2, 1)) + x_f.reshape(8, 1)
        opti.subject_to(xk_next == ca.reshape(xk_collocation, 1, 8))

    # Add input constraints
    input_max = params['input_max']
    input_min = params['input_min']
    for k in range(N-1):
        uk = U[k, :]
        for i in range(2):
            opti.subject_to(opti.bounded(input_min - u_f, uk[0, i], input_max - u_f))

    # Add boundary constraints
    # xmin, ymin, xmax, ymax = boundary
    print("boundary =", boundary)

    for k in range(N):
        # Extract the position state at timestep k
        xk = X[k, 0]
        yk = X[k, 1]

        # Add boundary constraints
        opti.subject_to(opti.bounded(xmin, xk, xmax))  # x-coordinate must be within boundaries
        opti.subject_to(opti.bounded(ymin, yk, ymax))  # y-coordinate must be within boundaries

    # Add top box obstacle constraints
    box = boxes[0]
    # xmin, ymin, xmax, ymax = box
    print("box =", box)
    # Define the margins around the box where the quadrotor should not enter
    # margin = 0  # Distance margin
    # for k in range(N):
    #     # Extract the position of the quadrotor at step k
    #     xk = X[k, 0]
    #     yk = X[k, 1]

    #     # Define the obstacle box with margin
    #     xmin_margin = xmin - margin
    #     ymin_margin = ymin - margin
    #     xmax_margin = xmax + margin

    #     # Constraints to keep the quadrotor outside the margin around the box
    #     outside_left = xk < xmin_margin
    #     outside_right = xk > xmax_margin
    #     outside_bottom = yk < ymin_margin

    #     # The quadrotor must be outside the margin around the top box
    #     opti.subject_to(outside_left + outside_right + outside_bottom >= 1)

    # Obstacle penalty constraint
    # barrier = 0
    # for k in range(N):
    #     xk, yk = X[k, 0], X[k, 1]
    #     # Check if inside the box
    #     inside_x_bounds = (xk > xmin) * (xk < xmax)
    #     inside_y_bounds = (yk > ymin) * (yk < ymax)
    #     inside_box = inside_x_bounds * inside_y_bounds

    #     barrier += inside_box * ((xk - xmin)**2 + (xk - xmax)**2 + (yk - ymin)**2 + (yk - ymax)**2)

    # Obstacle barrier function constraint
    # epsilon = 1e-3  # Small offset to prevent the log from blowing up
    # barrier = 0
    # for k in range(N):
    #     xk, yk = X[k, 0], X[k, 1]
    #     barrier += -ca.log(xk - xmin + epsilon)  # Barrier for left edge
    #     barrier += -ca.log(xmax - xk + epsilon)  # Barrier for right edge
    #     barrier += -ca.log(yk - ymin + epsilon)  # Barrier for bottom edge
    #     barrier += -ca.log(ymax - yk + epsilon)  # Barrier for top edge

    # Obstacle signed distance field
    # sdf = SignedDistanceField("./configs/world.yaml", lambda_param)
    # barrier = 0
    # for k in range(N-1):
    #     barrier += sdf.barrier_func(X[k, :])

    # Cost function on input
    cost = 0
    for k in range(N-1):
        cost += ca.sumsqr(U[k, :]) + alpha * ellipsoidal_function(X[k, :], box, lambda_param)

    opti.minimize(cost)
    # opti.minimize(cost + alpha * barrier)

    # Solve the optimization problem
    opts = {
        'ipopt.max_iter': opt_params['max_iter'],                # Maximum number of iterations
        'ipopt.acceptable_tol': opt_params['acceptable_tol'],          # Acceptable convergence tolerance
        'ipopt.acceptable_constr_viol_tol': opt_params['acceptable_constr_viol_tol']  # Acceptable constraint violation tolerance
    }

    opti.solver("ipopt", opts)
    try:
        sol = opti.solve()

        # Post-processing
        x_opt = sol.value(X)
        u_opt = sol.value(U)

        u_opt += u_f

        success = True
        return x_opt, u_opt, success
    
    except RuntimeError as e:
        print("Solver failed:", e)
        # Evaluate and print the value of a decision variable or expression
        print("Value of X:", opti.debug.value(X))
        print("Value of U:", opti.debug.value(U))

        success = False
        return opti.debug.value(X), opti.debug.value(U), success