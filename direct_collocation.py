import matplotlib.pyplot as plt
import numpy as np
import importlib

from MEAM517_Final_Project.quadrotor_with_pendulum import QuadrotorPendulum

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve, DirectCollocation
)

import kinematic_constraints
import dynamics_constraints
importlib.reload(kinematic_constraints)
importlib.reload(dynamics_constraints)
from kinematic_constraints import (
  AddFinalLandingPositionConstraint
)
from dynamics_constraints import (
  AddCollocationConstraints,
  EvaluateDynamics
)

def direct_collocation(N, x0, xf,):
    """
    
    """
    quadrotor = QuadrotorPendulum()

    context = quadrotor.CreateDefaultContext()

    num_t_step = 100
    min_t_step = 0.05
    max_t_step = 0.2

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

    dircol.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())

    dircol.AddBoundingBoxConstraint(xf, xf, dircol.final_state())

    dircol.AddRunningCost(dircol.input()[0]**2)
    dircol.AddRunningCost(dircol.input()[1]**2)



def window_constraint(quadrotor, x):
    """
    
    """
    end_pos = quadrotor.get_ends(x)
    
