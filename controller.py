import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def compute_heading(centerline: np.ndarray) -> np.ndarray:
    """
    Compute heading angle at each centerline point using finite differences.
    """
    diffs = np.diff(centerline, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.append(headings, headings[-1])


def compute_curvature(centerline: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Approximate curvature along the centerline:
        kappa = d(phi)/ds
    Smooths curvature to avoid noise.
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
    Cross-track error using RaceTrack boundaries – robust on any track.
    Returns:
        s    : index of closest centerline point
        e_ct : signed lateral deviation (>0 toward right boundary)
    """
    centerline = racetrack.centerline
    rb = racetrack.right_boundary
    lb = racetrack.left_boundary

    x, y = state[0], state[1]
    pos = np.array([x, y])

    # nearest centerline point
    d = np.linalg.norm(centerline - pos, axis=1)
    s = int(np.argmin(d))

    # normal from centerline to right boundary
    v = rb[s] - centerline[s]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-6:
        v = centerline[s] - lb[s]
        norm_v = np.linalg.norm(v)
    normal = v / (norm_v + 1e-6)

    # signed lateral error
    e_ct = np.dot(pos - centerline[s], normal)
    return s, e_ct


# ------------------------------------------------------------
# Low-level controller (unchanged)
# ------------------------------------------------------------

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
    # Cache heading and curvature along centerline
    # ------------------------------------------------------
    if not hasattr(racetrack, "_heading"):
        racetrack._heading = compute_heading(centerline)

    if not hasattr(racetrack, "_curvature"):
        racetrack._curvature = compute_curvature(centerline, racetrack._heading)

    heading = racetrack._heading
    curvature = racetrack._curvature

    # ------------------------------------------------------
    # Cross-track error + nearest centerline point
    # ------------------------------------------------------
    s, e_ct = compute_cross_track_error(state, racetrack)

    # Heading error
    phi_ref = heading[s]
    e_phi = np.arctan2(np.sin(phi_ref - phi), np.cos(phi_ref - phi))

    # ------------------------------------------------------
    # PID on heading
    # ------------------------------------------------------
    dt = 0.1

    if not hasattr(racetrack, "_phi_int"):
        racetrack._phi_int = 0.0
        racetrack._phi_prev = e_phi

    Kp = 0.6
    Ki = 0.05
    Kd = 0.1

    racetrack._phi_int += e_phi * dt
    racetrack._phi_int = np.clip(racetrack._phi_int, -0.4, 0.4)

    e_phi_dot = (e_phi - racetrack._phi_prev) / dt
    racetrack._phi_prev = e_phi

    delta_pid = Kp * e_phi + Ki * racetrack._phi_int + Kd * e_phi_dot

    # ------------------------------------------------------
    # Cross-track proportional correction
    # ------------------------------------------------------
    Kct = 0.05
    delta_ct = Kct * e_ct

    # ------------------------------------------------------
    # Total steering angle (direct command)
    # ------------------------------------------------------
    delta_r = delta_pid + delta_ct
    delta_r = np.clip(delta_r, delta_min, delta_max)

        # ------------------------------------------------------
    # Curvature-based slowdown (normalized)
    # ------------------------------------------------------
    kappa = curvature[s]

    # Cache max curvature to normalize [0, 1]
    if not hasattr(racetrack, "_kappa_max"):
        racetrack._kappa_max = np.max(np.abs(curvature)) + 1e-6  # avoid div0

    kappa_norm = np.clip(abs(kappa) / racetrack._kappa_max, 0.0, 1.0)

    # Map curvature to a target curve speed
    base_speed     = 22.0  # straight-line target speed
    min_turn_speed = 1.0  # slowest speed in the tightest turn

    v_curve = base_speed - kappa_norm * (base_speed - min_turn_speed)

    # ------------------------------------------------------
    # Add error-based slowdown on top of curvature-based speed
    # ------------------------------------------------------
    v_r = (
        v_curve
        - 3.0 * abs(e_phi)
        - 0.5 * abs(e_ct)
    )

    v_r = np.clip(v_r, 1.0, v_max)

    return np.array([delta_r, v_r])

