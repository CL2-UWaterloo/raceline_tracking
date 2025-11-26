import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

def compute_heading(centerline: np.ndarray) -> np.ndarray:
    """
    Compute heading angle at each centerline point using finite differences.

    centerline: array of points along the track
    """

    # calculate approximate segment vectors between consecutive points
    diffs = np.diff(centerline, axis=0)

    # each segment vector, compute a heading angle
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])

    # since we have one fewer diff element than points, we add headings[-1] to make return
    # array same length as center line
    return np.append(headings, headings[-1])


def compute_curvature(centerline: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Approximate curvature along the centerline:
        kappa = d(phi)/ds
    
        d(phi) = change in heading between points i-1 and i+1
        ds = change in distanc ebetween centerline[i-1] and centerline[i+1]
    """
    N = centerline.shape[0]
    kappa = np.zeros(N)

    for i in range(1, N - 1):
        dphi = heading[i+1] - heading[i-1]
        dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap correctly
        ds = np.linalg.norm(centerline[i+1] - centerline[i-1])
        kappa[i] = dphi / (ds + 1e-6)

    # Smooth curvature with a 9-point moving average
    kernel = np.ones(9) / 9.0
    return np.convolve(kappa, kernel, mode="same")


def compute_cross_track_error(state: ArrayLike, racetrack: RaceTrack):
    """
    Cross-track error using RaceTrack boundaries
    Returns:
        s    : index of closest centerline point
        e_ct : signed lateral deviation (>0 toward right boundary)
    """
    centerline = racetrack.centerline
    rb = racetrack.right_boundary
    lb = racetrack.left_boundary

    # get car pos
    x, y = state[0], state[1]
    pos = np.array([x, y])

    # find nearest centerline point to car
    d = np.linalg.norm(centerline - pos, axis=1)
    s = int(np.argmin(d)) # s is the index of the nearest centerline point

    # rb[s]: right boundary point
    # find the vector pointing from the centerline and the right edge
    v = rb[s] - centerline[s]
    
    # normalize v to unit length to only get direction
    norm_v = np.linalg.norm(v) 
    if norm_v < 1e-6:
        v = centerline[s] - lb[s]
        norm_v = np.linalg.norm(v)
    normal = v / (norm_v + 1e-6)

    # pos - centerline[s] - vector from centerline point to car
    # project this vector to the normal to see if car is towards left or right boundary
    # positive = car is towards right boundary
    # negative = car is towards left boundary
    # e_ct - how far sideways am I from track centerline
    e_ct = np.dot(pos - centerline[s], normal)
    return s, e_ct



def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    """
    desired = [delta_ref, v_ref]
    returns [steering_rate_cmd, accel_cmd]
    """
    delta = state[2]
    v = state[3]

    delta_r, v_r = desired

    # Steering rate control (P)
    k_delta = 3.0
    v_delta_cmd = k_delta * (delta_r - delta)

    # Velocity control (P)
    k_v = 1.0
    a_cmd = k_v * (v_r - v)

    return np.array([v_delta_cmd, a_cmd])


# ------------------------------------------------------------
# High-level curvature-aware controller
# ------------------------------------------------------------

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    x, y, delta, v, phi = state

    # Car parameters
    L         = parameters[0]   # wheelbase
    delta_min = parameters[1]
    delta_max = parameters[4]
    v_max     = parameters[5]

    centerline = racetrack.centerline

    # ------------------------------------------------------
    # Cache heading and curvature
    # ------------------------------------------------------
    if not hasattr(racetrack, "_heading"):
        racetrack._heading = compute_heading(centerline)

    if not hasattr(racetrack, "_curvature"):
        racetrack._curvature = compute_curvature(centerline, racetrack._heading)

    heading   = racetrack._heading
    curvature = racetrack._curvature

    # ------------------------------------------------------
    # Cross-track error + LOCAL heading error (no lookahead)
    # ------------------------------------------------------
    s, e_ct = compute_cross_track_error(state, racetrack)

    # desired heading at the same nearest centerline point
    phi_ref = heading[s]
    e_phi   = np.arctan2(np.sin(phi_ref - phi), np.cos(phi_ref - phi))

    heading_ahead = heading[(s + 8) % len(heading)]
    dphi_ahead = np.arctan2(np.sin(heading_ahead - heading[s]), 
                            np.cos(heading_ahead - heading[s]))

    sharpness = abs(dphi_ahead) / np.pi 

    # ------------------------------------------------------
    # Look-ahead curvature for severity (for gain + speed)
    # ------------------------------------------------------
    L_steps = int(1 +  v)    # dynamic look-ahead for curvature
    N = len(curvature)
    idx = np.arange(s, s + L_steps) % N

    kappa_ahead = np.max(np.abs(curvature[idx]))  # worst curve coming up

    if not hasattr(racetrack, "_kappa_max"):
        racetrack._kappa_max = np.max(np.abs(curvature)) + 1e-6
    kappa_norm = kappa_ahead / racetrack._kappa_max   # in [0,1]

    # mild gain factor (up to ~3x in very tight bits)
    gain_factor = 1.0 + 2.0 * kappa_norm

    # ------------------------------------------------------
    # PID on heading error (gain-scheduled, but local)
    # ------------------------------------------------------
    dt = 0.1
    if not hasattr(racetrack, "_phi_int"):
        racetrack._phi_int  = 0.0
        racetrack._phi_prev = e_phi

    # base gains
    Kp_base = 0.5
    Ki_base = 0.04
    Kd_base = 0.10

    # scale P and I slightly with curvature severity
    Kp = Kp_base * gain_factor
    Ki = Ki_base * gain_factor
    Kd = Kd_base  # derivative unscaled

    racetrack._phi_int += e_phi * dt
    racetrack._phi_int = np.clip(racetrack._phi_int, -0.4, 0.4)

    e_phi_dot = (e_phi - racetrack._phi_prev) / dt
    racetrack._phi_prev = e_phi

    delta_pid = Kp * e_phi + Ki * racetrack._phi_int + Kd * e_phi_dot

    # ------------------------------------------------------
    # Cross-track correction (also slightly stronger in tight corners)
    # ------------------------------------------------------
    Kct_base = 0.05
    Kct = Kct_base * (1.0 + 1.0 * kappa_norm)
    delta_ct = Kct * e_ct

    # ------------------------------------------------------
    # Curvature feedforward (using local curvature only)
    # ------------------------------------------------------
    kappa_here = curvature[s]
    Kff = 0.9
    delta_ff = Kff * np.arctan(L * kappa_here)

    # final steering: local heading + cross-track + local curvature FF
    delta_r = delta_ff + delta_pid + delta_ct
    delta_r = np.clip(delta_r, delta_min, delta_max)

    # ------------------------------------------------------
    # SPEED COMPUTATION USING LATERAL ACCEL LIMIT + LOOK-AHEAD
    # ------------------------------------------------------
    base_speed = 100.0
    min_turn_speed = -10.0
    a_lat_max = 8.0

    v_heading = base_speed - sharpness * (base_speed - 3)

    if kappa_ahead < 1e-6:
        v_curve = base_speed
    else:
        v_curve = np.sqrt(a_lat_max / (kappa_ahead + 1e-9))
        v_curve = np.clip(v_curve, min_turn_speed, base_speed)

    # stronger error-based slowdown in severe geometries
    err_gain = 1.0 + 10 * kappa_norm

    v_r = min(v_curve, v_heading)

    v_r -= err_gain * (3.0 * abs(e_phi) + 0.5 * abs(e_ct))


    v_r = np.clip(v_r, 1.0, v_max)

    return np.array([delta_r, v_r])


