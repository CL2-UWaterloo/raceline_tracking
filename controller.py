import numpy as np
from math import atan2, e, floor
from numpy.typing import ArrayLike

from simulator import RaceTrack

global iterator
iterator = 0

def lower_controller( # C_2 C_1
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:

    # Apply P(apply proportional const to current v) I(apply proportional const to position s_x) D(apply proportional const to acceleration)
    # Apply P(ditto to omega) I((v_r/l_wb)*K * heading, integral of omega) D(ditto to v_omega)

    # [steer angle, velocity]
    assert(desired.shape == (2,))

    # Control law for acceleration
    v_current = np.sqrt((state[2])**2 + (state[3])**2)

    a = 5.0 * (desired[1] - v_current)  # Proportional control
    
    # Process variable = delta
    K_p = 0.1
    K_i = 1
    K_d = 10
    heading_error = (desired[0] - state[4]) / 1
    # Normalize angle to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    P_control = desired[0] * K_p
    D_control = heading_error * K_d

    v_delta = P_control + D_control
    print("V_DELTA: " + str(v_delta))

    return np.array([a, v_delta]).T

def controller( # S_1 S_2
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
        # Current position

    global iterator
    iterator += 0.25
    x, y = state[0], state[1]
    v_x, v_y = state[2], state[3]
    psi = state[4]  # heading angle
    
    # Find closest point on raceline
    car_pos = np.array([x, y])
    distances = np.linalg.norm(racetrack.centerline - car_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # Look-ahead distance (tune this parameter)
    look_ahead = 2.0  # meters ahead on track
    
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
    dt = 0.01
    target_point = racetrack.centerline[floor(target_idx)]
    print("TARGET POINT: " + str(target_point))
    
    # Calculate desired velocity vector
    dx_des = (target_point[0] - x)
    dy_des = (target_point[1] - y)
    
    # Desired heading to target
    delta_desired = ((np.atan2(dy_des, dx_des) - state[2]) * parameters[0]) / (50 * dt)

    print("DELTA_DESIRED: " + str(delta_desired))
    
    # Desired speed (from raceline optimization or constant)
    test_const = -1
    v_change = (dy_des / dx_des) * test_const * 0
    print("V_CHANGE: " + str(v_change))
    v_desired = 50.0
    # v_desired = min((state[3] + v_change), parameters[9]) # m/s (tune based on track)
    
    return np.array([delta_desired, v_desired])