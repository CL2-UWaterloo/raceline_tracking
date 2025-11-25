import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from simulator import RaceTrack

# Helpers
def compute_heading(centerline: np.ndarray) -> np.ndarray:
    diffs = np.diff(centerline, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.append(headings, headings[-1])
    return headings


def compute_cross_track_error(state: ArrayLike, racetrack: RaceTrack):
    """
    Cross-track error using the RaceTrack's own left/right boundaries.
    Robust for arbitrary tracks and eliminates sign ambiguity issues.
    """
    centerline = racetrack.centerline
    right_boundary = racetrack.right_boundary
    left_boundary = racetrack.left_boundary

    x, y = state[0], state[1]
    pos = np.array([x, y])

    # 1. Nearest centerline index
    dists = np.linalg.norm(centerline - pos, axis=1)
    s = int(np.argmin(dists))

    # 2. Normal vector from centerline to right boundary
    v_right = right_boundary[s] - centerline[s]
    norm_v = np.linalg.norm(v_right)
    if norm_v < 1e-6:
        v_right = centerline[s] - left_boundary[s]
        norm_v = np.linalg.norm(v_right)

    n_right = v_right / (norm_v + 1e-6)

    # 3. Signed lateral distance
    e_ct = np.dot(pos - centerline[s], n_right)

    return s, e_ct


# ------------------------------------------------------------
# Low-level controller
# ------------------------------------------------------------

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:

    delta = state[2]
    v = state[3]

    delta_r = desired[0]
    v_r = desired[1]

    # Steering rate (P)
    k_delta = 3.0
    v_delta_cmd = k_delta * (delta_r - delta)

    # Velocity (P)
    k_v = 1.0
    a_cmd = k_v * (v_r - v)

    return np.array([v_delta_cmd, a_cmd])


# ------------------------------------------------------------
# High-level controller
# ------------------------------------------------------------

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:

    sx, sy, delta, v, phi = state
    centerline = racetrack.centerline
    # Cache heading
    if not hasattr(racetrack, "_heading"):
        racetrack._heading = compute_heading(centerline)
    heading = racetrack._heading

    # One consistent projection for heading + cross-track
    s, e_ct = compute_cross_track_error(state, racetrack)
    phi_ref = heading[s]

    # Heading error
    e_phi = np.arctan2(np.sin(phi_ref - phi), np.cos(phi_ref - phi))

    # Initialize integrals
    dt = 0.1

    if not hasattr(racetrack, "_phi_int"):
        racetrack._phi_int = 0.0
    if not hasattr(racetrack, "_phi_prev"):
        racetrack._phi_prev = e_phi

    if not hasattr(racetrack, "_ct_int"):
        racetrack._ct_int = 0.0

    # PID on heading error
    Kp = 0.5
    Ki = 0.1
    Kd = 0.1

    racetrack._phi_int += e_phi * dt
    racetrack._phi_int = np.clip(racetrack._phi_int, -0.5, 0.5)  # anti-windup

    e_phi_dot = (e_phi - racetrack._phi_prev) / dt
    racetrack._phi_prev = e_phi

    delta_pid = Kp * e_phi + Ki * racetrack._phi_int + Kd * e_phi_dot

    # Cross-track PID (with filtered derivative)
    Kct_p = 0.06
    Kct_i = 0.005
    Kct_d = 0.02 

    # integral (anti-windup)
    racetrack._ct_int += e_ct * dt
    racetrack._ct_int = np.clip(racetrack._ct_int, -0.4, 0.4)

    # filtered derivative
    if not hasattr(racetrack, "_ct_prev"):
        racetrack._ct_prev = e_ct
    if not hasattr(racetrack, "_ct_d_filtered"):
        racetrack._ct_d_filtered = 0.0

    raw_d = (e_ct - racetrack._ct_prev) / dt
    racetrack._ct_prev = e_ct

    # first-order lowpass filter to remove noise
    alpha = 0.2
    racetrack._ct_d_filtered = (1 - alpha) * racetrack._ct_d_filtered + alpha * raw_d

    # final cross-track PID term
    delta_ct = (
        Kct_p * e_ct +
        Kct_i * racetrack._ct_int +
        Kct_d * racetrack._ct_d_filtered
    )


    # Total steering
    delta_r = delta_pid + delta_ct
    delta_r = np.clip(delta_r, -0.4, 0.4)

    # Speed control
    base_speed = 12.0
    v_r = base_speed - 4.0 * abs(e_phi) - 0.7 * abs(e_ct)
    v_r = np.clip(v_r, 5.0, 20.0)

    return np.array([delta_r, v_r])
