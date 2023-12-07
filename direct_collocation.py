import numpy as np
import importlib
from obstacles import Obstacles

from quadrotor_with_pendulum import QuadrotorPendulum

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve, DirectCollocation
)

# import kinematic_constraints
# import dynamics_constraints
# importlib.reload(kinematic_constraints)
# importlib.reload(dynamics_constraints)
# from kinematic_constraints import (
#     AddFinalLandingPositionConstraint
# )
# from dynamics_constraints import (
#     AddCollocationConstraints,
#     EvaluateDynamics
# )

def direct_collocation(quadrotor, obstacles, N, x_i, u_i, dt):
    """
    Given quadrotor model, obstacles, number of collocation points, initial trajectory,
    and 
    
    """
    context = quadrotor.CreateDefaultContext()
    boundary, boxes = obstacles.get_world()

    print('context = ', context)

    num_t_step = N
    min_t_step = dt
    max_t_step = dt

    dircol = DirectCollocation(quadrotor, context, num_t_step, min_t_step, max_t_step)

    dircol.AddEqualTimeIntervalsConstraints()

    max_input = 30
    min_input = 0

    # Constraint for the first component of the control input
    dircol.AddConstraintToAllKnotPoints(dircol.input()[0] <= max_input)
    dircol.AddConstraintToAllKnotPoints(dircol.input()[0] >= min_input)

    # Constraint for the second component of the control input
    dircol.AddConstraintToAllKnotPoints(dircol.input()[1] <= max_input)
    dircol.AddConstraintToAllKnotPoints(dircol.input()[1] >= min_input)

    # Set initial and final state constraints
    dircol.AddBoundingBoxConstraint(x_i[0], x_i[0], dircol.initial_state())
    dircol.AddBoundingBoxConstraint(x_i[-1], x_i[-1], dircol.final_state())

    # Add running cost on inputs
    dircol.AddRunningCost(dircol.input()[0]**2)
    dircol.AddRunningCost(dircol.input()[1]**2)

    # Window constraint
    box_1 = boxes[0]
    for i in range(N):
        state = dircol.state(i)
        dircol.AddConstraint(box_constraint(quadrotor, state, box_1) <= 0)

    box_2 = boxes[1]
    for i in range(N):
        state = dircol.state(i)
        dircol.AddConstraint(box_constraint(quadrotor, state, box_2) <= 0)

    # Boundary constraint
    xmin, ymin, xmax, ymax = boundary
    for i in range(N):
        # Access the state at the i-th collocation point
        state = dircol.state(i)
        
        # Extract the position variables from the state
        x = state[0]
        y = state[1]
        
        # Apply the bounding box constraint
        dircol.AddBoundingBoxConstraint([xmin, ymin], [xmax, ymax], [x, y])

    # Set initial guess
    for i in range(N):
        dircol.SetInitialGuess(dircol.state(i), x_i[i])
        dircol.SetInitialGuess(dircol.input(i), u_i[i])

    result = Solve(dircol)
    assert result.is_success()

    # Get optimized trajectory
    optimized_trajectory = dircol.ReconstructInputTrajectory(result)

    return optimized_trajectory
    
    
def box_constraint(quadrotor, state, box):
    """
    Returns 1 if the quadrotor collides a box, 0 otherwise
    
    """
    xb = state[0]
    yb = state[1]
    xr, yr, xl, yl, xm, ym = quadrotor.get_ends(x)

    x_min, y_min, x_max, y_max = box
    # Apply window constraint only if the quadrotor is within the x-axis range of the window
    if (x_min <= xr <= x_max) or (x_min <= xl <= x_max) or (x_min <= xm <= x_max):
        # If quadrotor is above the obstacle
        if yb > y_max:
            if (yr < y_max) and (yl < y_max) and (ym < y_max):
                return 1
        # If quadrotor is below the obstacle
        elif yb < y_min:
            if (yr > y_min) and (yl > y_min) and (ym > y_min):
                return 1

    return 0  # Constraint is satisfied if outside the x-axis range of the window
    

if __name__ == '__main__':
    quadrotor = QuadrotorPendulum()
    obstacles = Obstacles()
    # optimized_trajectory = direct_collocation(quadrotor, obstacles, N, x0, xf)