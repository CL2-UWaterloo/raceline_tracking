import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:

    # [steer angle, velocity]
    assert(desired.shape == (2,))

    return np.array([0, 100]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    # Set desired v to max if steering angle is 0, else set to min
        # TEST RUN: a = a_max always, only direction changes

    # To get desired steering rate, find change in heading rate based on desired steering angle (from track),
    
    # v_r \approx ((s_x at index i+1) - (s_x at index i))/t_omega
    # phi \approx ((heading at index i+1) - (heading at index i))/t_omega
    # omega_r \approx (phi * l_wb) / v_r
    # v_omega \approx min(|(omega_r - omega)|/1 second, v_omega max)

    # if omega_r == 0:
        # a = a_max
    # else:
        # a = -a_max


    return np.array([0, 100]).T