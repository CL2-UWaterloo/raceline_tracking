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

    # compute heading and curvature and cache them
    if not hasattr(racetrack, "_heading"):
        racetrack._heading = compute_heading(centerline)

    if not hasattr(racetrack, "_curvature"):
        racetrack._curvature = compute_curvature(centerline, racetrack._heading)

    heading = racetrack._heading
    curvature = racetrack._curvature

    # compute cross track error
    # s - index of nearest track point
    # e_ct - sideway distance from centerline
    s, e_ct = compute_cross_track_error(state, racetrack)

    rb = racetrack.right_boundary
    lb = racetrack.left_boundary
    track_width      = np.linalg.norm(rb[s] - lb[s])
    track_half_width = 0.5 * track_width

    # use a stricter “off track” definition
    off_track = abs(e_ct) > track_half_width

    # ------------------------------------------------------
    # Heading error (difference from path heading)
    # ------------------------------------------------------
    phi_ref = heading[s]
    e_phi = np.arctan2(np.sin(phi_ref - phi), np.cos(phi_ref - phi))

    dt = 0.1
    if not hasattr(racetrack, "_phi_int"):
        racetrack._phi_int = 0.0
        racetrack._phi_prev = e_phi

    Kp = 0.7
    Ki = 0.03
    Kd = 0.12

    racetrack._phi_int += e_phi * dt
    racetrack._phi_int = np.clip(racetrack._phi_int, -0.4, 0.4)

    e_phi_dot = (e_phi - racetrack._phi_prev) / dt
    racetrack._phi_prev = e_phi

    delta_pid = Kp * e_phi + Ki * racetrack._phi_int + Kd * e_phi_dot

    if not hasattr(racetrack, "_e_ct_prev"):
        racetrack._e_ct_prev = e_ct

    e_ct_dot = (e_ct - racetrack._e_ct_prev) / dt
    racetrack._e_ct_prev = e_ct

    Kct   = 0.04     # P on cross-track
    Kct_d = 0.02     # D on cross-track

    delta_ct = Kct * e_ct + Kct_d * e_ct_dot
    delta_ct = np.clip(delta_ct, -0.18, 0.18)

    # ------------------------------------------------------
    # Total steering angle (direct command)
    # ------------------------------------------------------
    delta_r = delta_pid + delta_ct
    delta_r = np.clip(delta_r, delta_min, delta_max)

    # ------------------------------------------------------
    # Curvature-based slowdown (normalized)
    # ------------------------------------------------------
    # window of points ahead to inspect curvature
    window = 30   # try 20–40 depending on track density
    s_end = min(s + window, len(curvature) - 1)
    kappa_window = np.abs(curvature[s:s_end+1])
    kappa_eff = np.max(kappa_window)  # “worst” curve coming up

    # cache max curvature for normalization
    if not hasattr(racetrack, "_kappa_max"):
        racetrack._kappa_max = np.max(np.abs(curvature)) + 1e-6

    kappa_norm = np.clip(kappa_eff / racetrack._kappa_max, 0.0, 1.0)

    # base speed profile:
    #   - high speed on straights
    #   - low speed in tight curves
    base_speed_straight = min(v_max, 32.0)  # fast on straights
    min_turn_speed      = 8.0               # tight corners

    # use an exponent > 1 for more aggressive slowdown in tight corners
    v_curve = min_turn_speed + (base_speed_straight - min_turn_speed) * (1.0 - kappa_norm)**1.8

    # ------------------------------------------------------
    # Error-based slowdown
    #   - big heading error -> slow down
    #   - large lateral offset -> slow down a lot
    # ------------------------------------------------------
    # lateral error as fraction of half-width (0 = center, 1 = at edge)
    e_lat_ratio = abs(e_ct) / (track_half_width + 1e-3)
    e_lat_ratio = np.clip(e_lat_ratio, 0.0, 2.0)

    v_r = v_curve

    # heading error penalty
    v_r -= 4.0 * abs(e_phi)

    # extra penalty for being off-center (scaled)
    v_r -= 6.0 * e_lat_ratio

    # clamp to physical max
    v_r = np.clip(v_r, 2.0, v_max)

    # ------------------------------------------------------
    # Safety clamps near edges / off track
    # ------------------------------------------------------
    # when near edge (but not fully off), cap speed
    if e_lat_ratio > 0.7:
        v_r = min(v_r, 10.0)

    # clearly off the track: go very slow to re-enter safely
    if off_track:
        v_r = 3.0

    # ------------------------------------------------------
    # Extra push on very straight, well-aligned segments
    #   -> more acceleration on safe straights
    # ------------------------------------------------------
    if (kappa_eff < 0.1 * racetrack._kappa_max   # almost straight ahead
        and e_lat_ratio < 0.3                    # near center
        and abs(e_phi) < 0.08                    # well aligned
        and not off_track):

        # allow extra speed boost on straights, but not beyond v_max
        v_r = min(v_max, v_r + 4.0)

    return np.array([delta_r, v_r])
