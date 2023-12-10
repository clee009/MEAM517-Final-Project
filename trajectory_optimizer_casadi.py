import casadi as ca
import numpy as np

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
    A_c, B_c = get_linearized_dynamics(x_f, u_f, params)

    # Identity matrix in CasADi
    I = ca.MX.eye(8)

    # Discretize the linearized dynamics
    A_d = I + A_c * dt
    B_d = B_c * dt

    return A_d, B_d

def optimize_trajectory(quadrotor, obstacles, N, dt, initial_trajectory):
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

    x_f = quadrotor.x_f
    u_f = quadrotor.u_f

    # Start optimization problem
    opti = ca.Opti()

    # Define decision variables for states and control inputs
    X = opti.variable(N, 8)  # State trajectory
    U = opti.variable(N-1, 2)   # Control inputs

    # Define initial guesses and initial state
    initial_state_guess = initial_trajectory['state']
    initial_control_guess = initial_trajectory['input']
    initial_state = initial_state_guess[0, :]
    print('intial_state shape =', initial_state.shape)
    print('U[0] shape =', U[0, :].shape)
    final_state = initial_state_guess[-1, :]

    # Set initial guesses
    opti.set_initial(X, initial_state_guess)
    opti.set_initial(U, initial_control_guess)

    # Add initial state constraint
    opti.subject_to(X[0, :] == initial_state)

    # Add final state constraint
    opti.subject_to(X[N-1, :] == final_state)

    # Add dynamics constraints
    A, B = discrete_time_linearized_dynamics(dt, x_f, u_f, params)
    for k in range(N-1):
        xk = X[k, :]
        xk_next = X[k+1, :]
        uk = U[k, :]
        xk_collocation = ca.mtimes(A, (xk - x_f)) + ca.mtimes(B, uk) + x_f
        opti.subject_to(xk_next == xk_collocation)

    # Add input constraints
    input_max = params['input_max']
    input_min = params['input_min']
    for k in range(N-1):
        uk = U[k, :]
        for ui in uk:
            opti.subject_to(opti.bounded(input_min, ui, input_max))

    # Add boundary constraints
    boundary = obstacles[0]

    for k in range(N):
        # Extract the position state at timestep k
        xk = X[k, 0]
        yk = X[k, 1]

        # Add boundary constraints
        opti.subject_to(opti.bounded(xmin, xk, xmax))  # x-coordinate must be within boundaries
        opti.subject_to(opti.bounded(ymin, yk, ymax))  # y-coordinate must be within boundaries

    # Add top box obstacle constraints
    box = obstacles[2]
    xmin, ymin, xmax, ymax = box
    # Define the margins around the box where the quadrotor should not enter
    margin = 0.5  # Distance margin
    for k in range(N):
        # Extract the position of the quadrotor at step k
        xk = X[k, 0]
        yk = X[k, 1]

        # Define the obstacle box with margin
        xmin_margin = xmin - margin
        ymin_margin = ymin - margin
        xmax_margin = xmax + margin

        # Constraints to keep the quadrotor outside the margin around the box
        outside_left = xk < xmin_margin
        outside_right = xk > xmax_margin
        outside_bottom = yk < ymin_margin

        # The quadrotor must be outside the margin around the top box
        opti.subject_to(outside_left + outside_right + outside_bottom >= 1)

    # Cost function on input
    cost = 0
    for k in range(N):
        cost += ca.sumsqr(U[k, :])

    opti.minimize(cost)

    # Solve the optimization problem
    opti.solver("ipopt")
    sol = opti.solve()

    # Post-processing
    x_opt = sol.value(X)
    u_opt = sol.value(U)

    return x_opt, u_opt