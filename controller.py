import numpy as np
from math import atan2, e, floor, sqrt
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
    v_current = state[3] #np.sqrt((state[2])**2 + (state[3])**2)

    a = 5 * (desired[1] - v_current)  # Proportional control
    if a < 0:
        a = max(a, parameters[8])
    else:
        a = min(a, parameters[10])

    # Once desired[0] is a certain level of positive, it starts cradlng
    
    # Process variable = delta
    K_p = 0.1
    K_i = 1
    K_d = 15
    heading_error = ((desired[0] - (state[4] + state[2])) * parameters[0]) / (50 * 0.1)
    # Normalize angle to [-pi, pi]
    heading_error = atan2(np.sin(heading_error), np.cos(heading_error))
    P_control = desired[0] * K_p
    D_control = heading_error * K_d

    v_delta = P_control + D_control
    print("V_DELTA: " + str(v_delta))

    return np.array([v_delta, a]).T

global prev_dist
prev_dist = 0

def controller( # S_1 S_2
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
        # Current position

    x, y = state[0], state[1]
    psi = state[2]
    v = state[3]
    current_delta = state[4]
    
    # Find closest point on raceline
    global iterator
    car_pos = np.array([x, y])
    distances = np.linalg.norm(racetrack.centerline - car_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # Look-ahead distance (tune this parameter)
    look_ahead = 3.0  # meters ahead on track
    
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

    # dist_const = 7
    # if ( np.linalg.norm(racetrack.centerline[target_idx] - car_pos) < dist_const ):
    # iterator = closest_idx
    
    # Desired position after one cycle (dt = 0.001s from simulator)
    dt = 0.01
    target_point = racetrack.centerline[target_idx]
    print("TARGET POINT: " + str(target_point))
    
    # Calculate desired velocity vector
    dx_des = (target_point[0] - x)
    dy_des = (target_point[1] - y)
    
    # Desired heading to target
    print("STATE ANGLE: " + str(state[4]))
    print("ATAN: " + str(atan2(dy_des, dx_des)))
    delta_desired = atan2(dy_des, dx_des)

    print("DELTA_DESIRED: " + str(delta_desired))
    
    # Desired speed (from raceline optimization or constant)
    global prev_dist
    v_change_const = 2
    v_change = sqrt(dy_des**2 + dx_des**2) * v_change_const
    print("HEADING: " + str(state[2]))
    print("V_CHANGE: " + str(v_change))
    v_desired = (state[3] + v_change)
    if v_desired < parameters[2] or abs(state[2]) > 0.15:
        v_desired = 1
    elif abs(state[2]) > 0.12:
        v_desired = 3
    elif abs(state[2]) < 0.05 and abs(state[2]) > 0.02:
        v_desired = 15
    elif abs(state[2]) < 0.08 and abs(state[2]) > 0.05:
        v_desired = 5
    elif v_desired > min(30, parameters[5]):
        v_desired = min(30, parameters[5])
    print(v_desired)
    
    return np.array([delta_desired, v_desired])