import numpy as np
from math import atan2
from numpy.typing import ArrayLike

from simulator import RaceTrack

def lower_controller( # C_2 C_1
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:

    # Apply P(apply proportional const to current v) I(apply proportional const to position s_x) D(apply proportional const to acceleration)
    # Apply P(ditto to omega) I((v_r/l_wb)*K * heading, integral of omega) D(ditto to v_omega)

    # [steer angle, velocity]
    assert(desired.shape == (2,))

    # Control law for acceleration
    v_current = np.sqrt(state[2]**2 + state[3]**2)

    a = 5.0 * (desired[1] - v_current)  # Proportional control
    
    # Control law for angular velocity (heading control)
    heading_error = desired[0] - state[4]
    # Normalize angle to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    v_delta = 2.0 * heading_error  # Proportional control

    return np.array([v_delta, a]).T

def controller( # S_1 S_2
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
        # Current position
    x, y = state[0], state[1]
    v_x, v_y = state[2], state[3]
    psi = state[4]  # heading angle
    
    # Find closest point on raceline
    car_pos = np.array([x, y])
    distances = np.linalg.norm(racetrack.centerline - car_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # Look-ahead distance (tune this parameter)
    look_ahead = 5.0  # meters ahead on track
    
    # Find target point ahead on raceline
    cumulative_dist = 0
    target_idx = closest_idx
    
    for i in range(closest_idx, len(racetrack.centerline)):
        if i == closest_idx:
            continue
        cumulative_dist += np.linalg.norm(racetrack.centerline[i] - racetrack.centerline[i-1])
        if cumulative_dist >= look_ahead:
            target_idx = i
            break
    
    # Desired position after one cycle (dt = 0.001s from simulator)
    dt = 0.001
    target_point = racetrack.centerline[target_idx]
    
    # Calculate desired velocity vector
    dx_des = target_point[0] - x
    dy_des = target_point[1] - y
    
    # Desired heading to target
    delta_desired = np.arctan2(dy_des, dx_des)
    
    # Desired speed (from raceline optimization or constant)
    v_desired = 20.0  # m/s (tune based on track)
    
    return np.array([delta_desired, v_desired])

    # CAN TREAT DISCRETE SYSTEM AS CONTINUOUS BECAUSE WE HAVE UNDERLYING DIFF EQN
    # Dist can be calc'd by speed and sampling speed
    # Set desired v to max if steering angle is 0, else set to min
        # TEST RUN: a = a_max always, only direction changes

    # To get desired steering rate, find change in heading rate based on desired steering angle (from track),

    # v_r = state[3]
    # if v_r == 0:
    #     v_r = 1
    
    # min_distance = np.argmin( np.linalg.norm( rt.centerline - np.array([state[0], state[1]]), axis=1 ) )

    # delta = ((atan2( state[1],  state[0]) - state[2])*parameters[0]) / (1 * v_r)

    # v_r \approx ((s_x at index i+1) - (s_x at index i))/t_omega
    # phi \approx ((heading at index i+1) - (heading at index i))/t_omega
    # omega_r \approx (phi * l_wb) / v_r
    # v_omega \approx min(|(omega_r - omega)|/1 second, v_omega max)

    # if omega_r == 0:
        # a = a_max
    # else:
        # a = -a_max


    # return np.array([delta, v_r]).T