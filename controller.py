import numpy as np
from numpy.typing import ArrayLike

# REMOVE PLT debug
from simulator import RaceTrack, plt

# state = [x, y, steering_angle, velocity, heading]



error_v_sum = 0
error_phi_sum = 0

last_error_v = None
last_error_phi = 0
dt = 0.1
def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    global error_v_sum, error_phi_sum, last_error_v, last_error_phi

    assert desired.shape == (2,)

    # --- Compute errors ---
    error_phi = desired[0] - state[2]
    error_v = desired[1] - state[3]

    # === Velocity PID ===
    Kp_v = 250
    Ki_v = 0
    Kd_v = 0

    error_v_sum += dt * error_v

    accel_cmd = Kp_v * error_v + Ki_v * error_v_sum
    accel_cmd = np.clip(accel_cmd, parameters[8], parameters[10])

    last_error_v = error_v

    # === Steering PID ===
    Kp_s = 10.0
    Ki_s = 0
    Kd_s = 2

    error_phi_sum += dt * error_phi

    steer_cmd = Kp_s * error_phi + Ki_s * error_phi_sum + Kd_s * (error_phi - last_error_phi)
    steer_cmd = np.clip(steer_cmd, parameters[7], parameters[9])

    last_error_phi = error_phi

    return np.array([steer_cmd, accel_cmd])



# global state
i = 0 # current index



def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    global i

    sx, sy = state[0], state[1]
    heading = state[4]

    points = racetrack.raceline
    lwb = parameters[0]

    # --- 1. Advance to nearest lookahead point ---
    distance_threshold = max(8.0, 0.25 * state[3])   # dynamic lookahead
    rx, ry = points[i]
    
    while (sx - rx)**2 + (sy - ry)**2 <= distance_threshold**2:
        i = (i + 1) % len(points)
        rx, ry = points[i]
    plt.plot(rx, ry, "o")
    dx = rx - sx
    dy = ry - sy

    # --- 2. Pure pursuit steering ---
    alpha = np.arctan2(dy, dx) - heading
    lookahead_dist = np.hypot(dx, dy)
    lookahead_dist = max(3.0, lookahead_dist)

    desired_angle = np.arctan2(2 * lwb * np.sin(alpha), lookahead_dist)
    desired_angle = np.clip(desired_angle, parameters[1], parameters[4])

    # --- 3. CURVATURE-BASED SPEED CONTROL ---
    # pick 3 points ahead: current, mid, far
    idx_mid = (i + 6) % len(points)
    idx_far = (i + 12) % len(points)
    p1 = np.array(points[i])
    p2 = np.array(points[idx_mid])
    p3 = np.array(points[idx_far])

    plt.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], "o")
    # compute curvature κ from circumcircle radius
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    # Heron's formula for triangle area
    s = 0.5 * (a + b + c)
    area = max(1e-6, np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c))))

    # curvature κ = 4A / (abc)
    curvature = 4.0 * area / max(1e-6, a * b * c)

    # limit curvature
    curvature = min(curvature, 0.5)  

    # --- Desired velocity based on curvature ---
    # max lateral accel (racecar ≈ 1.8 g)
    a_lat_max = 17.7

    if curvature < 1e-4:
        v_curve = parameters[5]    # straight line → max velocity
    else:
        v_curve = np.sqrt(a_lat_max / curvature)

    # Clip using car physical limits
    desired_velocity = np.clip(v_curve, parameters[2], parameters[5])

    return np.array([desired_angle, desired_velocity])

