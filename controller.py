import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

from blocks import S1, S2
from helpers import estimate_upcoming_curvature


class C1:
    def __init__(self):
        # Parameters for velocity PD controller
        self.K_p = 2.0
        self.K_d = 0.1

    def step(
        self, state: np.ndarray, parameters: np.ndarray, desired_velocity: float
    ) -> float:
        """
        A simple PID controller, outputs an acceleration command to reach desired velocity
        """
        current_velocity = state[3]
        error = desired_velocity - current_velocity

        a = self.K_p * error

        # TODO: Add derivative terms
        max_acceleration = parameters[10]
        return np.clip(a, -max_acceleration, max_acceleration)


class C2:
    def __init__(self):
        # Parameters for steering Pd controller
        self.K_p = 3.0
        self.K_d = 0.5

    def step(
        self, state: np.ndarray, parameters: np.ndarray, desired_steering: float
    ) -> float:
        """
        A simple PD controller, outputs a steering rate command to reach desired steering angle
        """
        current_steering = state[2]
        steering_error = desired_steering - current_steering

        steering_rate = self.K_p * steering_error

        # TODO: Add derivative terms
        max_steering_vel = parameters[9]
        return np.clip(steering_rate, -max_steering_vel, max_steering_vel)


# Create instances of blocks
c1 = C1()
c2 = C2()

s1 = S1()
s2 = S2()


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Lower-level controller that converts desired steering angle and velocity
    to steering rate and acceleration commands.

    Args:
        state: Current state [x, y, steering_angle, velocity, heading]
        desired: Desired [steering_angle, velocity]
        parameters: Vehicle parameters

    Returns:
        Control inputs [steering_rate, acceleration]
    """
    assert desired.shape == (2,)

    steering_rate_command = c2.step(state, parameters, desired[0])
    acceleration_command = c1.step(state, parameters, desired[1])

    return np.array([steering_rate_command, acceleration_command]).T


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    # Vehicle parameters
    wheelbase = parameters[0]
    current_velocity = state[3]

    # Dynamic lookahead distance based on velocity AND upcoming curvature
    # Estimate upcoming path curvature
    upcoming_curvature = estimate_upcoming_curvature(
        state, racetrack.centerline, preview_points=15
    )

    # Base lookahead from velocity (higher speed = look further)
    base_lookahead = 12.0 + current_velocity * 0.4

    # Curvature adjustment: high curvature ahead = increase lookahead to prepare earlier
    # Low curvature (straight) = can use shorter lookahead
    if upcoming_curvature > 0.01:  # Significant curve ahead
        # Increase lookahead proportionally to curvature (more curve = look further)
        curvature_factor = 1.0 + min(
            upcoming_curvature * 20.0, 1.5
        )  # Up to 2.5x increase
    else:  # Straight section
        curvature_factor = 0.9  # Slightly reduce for efficiency

    lookahead_distance = base_lookahead * curvature_factor
    lookahead_distance = np.clip(
        lookahead_distance, 8.0, 60.0
    )  # Increased max for tight corners

    desired_steering = s2.step(
        state, racetrack.centerline, lookahead_distance, wheelbase
    )
    desired_velocity = s1.step(state, racetrack.centerline)

    return np.array([desired_steering, desired_velocity]).T
