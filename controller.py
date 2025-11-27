import numpy as np
from numpy.typing import ArrayLike

# REMOVE PLT debug
from simulator import RaceTrack, plt

# state = [x, y, steering_angle, velocity, heading]



# GLOBAL CONTROLLER PARAMS





error_v_sum = 0
error_phi_sum = 0

last_error_v = 0 
last_error_phi = 0
dt = 0.1
def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    global error_v_sum, error_phi_sum, last_error_v, last_error_phi
    assert desired.shape == (2,)


    Kp_v = 200
    Ki_v = 0
    Kd_v = 0

    Kp_s = 10.0
    Ki_s = 0
    Kd_s = 2.0


    error_v = desired[1] - state[3]
    error_v_sum += dt * error_v

    accel_cmd = Kp_v * error_v + Ki_v * error_v_sum + Kd_v * (error_v - last_error_v)
    accel_cmd = np.clip(accel_cmd, parameters[8], parameters[10])

    last_error_v = error_v

    error_phi = desired[0] - state[2]
    error_phi_sum += dt * error_phi

    steer_cmd = Kp_s * error_phi + Ki_s * error_phi_sum + Kd_s * (error_phi - last_error_phi)
    steer_cmd = np.clip(steer_cmd, parameters[7], parameters[9])

    last_error_phi = error_phi

    return np.array([steer_cmd, accel_cmd])



# global state
i = 0 # current index

def reset_globals():
    global i, error_v_sum, error_phi_sum, last_error_v, last_error_phi
    i = 0
    error_v_sum = 0
    error_phi_sum = 0

    last_error_v = 0
    last_error_phi = 0



def closest_point_on_segment(rx, ry, px, py, sx, sy):
    # Vector R→P
    vx = px - rx
    vy = py - ry

    # Vector R→S
    wx = sx - rx
    wy = sy - ry

    # Compute projection t of w onto v (normalized by |v|^2)
    segment_len_sq = vx*vx + vy*vy
    if segment_len_sq == 0:
        # R and P are the same point
        return rx, ry

    t = (wx*vx + wy*vy) / segment_len_sq

    # Clamp t to segment [0, 1]
    t = max(0, min(1, t))

    # Closest point = R + t·(P−R)
    cx = rx + t * vx
    cy = ry + t * vy

    return cx, cy

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    global i

    sx, sy = state[0], state[1]
    heading = state[4]

    points = racetrack.raceline
    lwb = parameters[0]

    # find lookahead
    distance_threshold = 18
    rx, ry = points[i]
    
    while (sx - rx)**2 + (sy - ry)**2 <= distance_threshold**2:
        i = (i + 1) % len(points)
        rx, ry = points[i]
    plt.plot(rx, ry, "o")
    
    rx, ry = closest_point_on_segment(rx, ry, points[i - 1][0], points[i - 1][1], sx, sy)
    dx = rx - sx
    dy = ry - sy


    # angle desired pure pursuit
    alpha = np.arctan2(dy, dx) - heading
    lookahead_dist = np.hypot(dx, dy)
    lookahead_dist = max(3.0, lookahead_dist)

    desired_angle = np.arctan2(2 * lwb * np.sin(alpha), lookahead_dist)
    desired_angle = np.clip(desired_angle, parameters[1], parameters[4])
    

    max_curv = 0.0

    # number of future segments to scan for curvature
    N = 43

    spacing = 3
    backwardsOffset = 1
    for k in range(-backwardsOffset, N - backwardsOffset):

        p1 = np.array(points[(i + k) % len(points)])
        p2 = np.array(points[(i + k + spacing) % len(points)])
        p3 = np.array(points[(i + k + spacing * 2) % len(points)])
        # distances between points
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # prevent numerical issues
        if a < 1e-4 or b < 1e-4 or c < 1e-4:
            continue

        s = 0.5 * (a + b + c)
        area = max(1e-6, np.sqrt(max(0, s*(s-a)*(s-b)*(s-c))))

        curvature = 4 * area / (a*b*c)
        # importance scaling
        scaling = 1 - ( k + backwardsOffset + 1 ) / N
        curvature = scaling * curvature
        max_curv = max(max_curv, curvature)

    curvature = max_curv


    a_lat_max = 60
    if curvature < 1e-4:
        v_curve = parameters[5]   
    else:
        v_curve = np.sqrt(a_lat_max / curvature)

    max_v_prop = 1
    min_target_prop = 0.3
    angle_prop =  np.abs(desired_angle) / parameters[4]
    angle_multiplier = 10
    offset = 0
    power = 3
    temp_v = parameters[5] * max(
        min_target_prop, 
        max_v_prop * (1.0 - (angle_multiplier * angle_prop + offset) ** power + offset ** power))

    # Clip using car physical limits
    desired_velocity = np.clip(min(v_curve, temp_v), parameters[2], parameters[5])

    return np.array([desired_angle, desired_velocity])

