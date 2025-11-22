import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

from blocks import S1


class C1:
    def __init__(self):
        self.K_p = 1.0
        self.K_i = 0.0
        self.K_d = 0.1

    def step(self, state: np.ndarray, desired: float) -> float:
        """
        A simple PID controller, outputs an acceleration command to reach desired velocity
        """
        current_velocity = state[3]
        error = desired - current_velocity

        a = self.K_p * error

        # TODO: Add integral and derivative terms
        return np.clip(a, -20, 20)


class C2:
    def __init__(self):
        self.K_p = 1.0


# Create instances of blocks
c1 = C1()
c2 = C2()

s1 = S1()


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    # [steer angle, velocity]
    assert desired.shape == (2,)

    acceleration_command = c1.step(state, desired[1])

    return np.array([0, acceleration_command]).T


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    desired_velocity = s1.step(state, racetrack.centerline)

    return np.array([0, desired_velocity]).T
