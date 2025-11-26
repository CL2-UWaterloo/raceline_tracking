import numpy as np
from numpy.typing import ArrayLike

# REMOVE PLT debug
from simulator import RaceTrack, plt

# state = [x, y, steering_angle, velocity, heading]
# paramters = self.parameters = np.array([
        #     self.wheelbase, # Car Wheelbase 0
        #     -self.max_steering_angle, # x3 1 
        #     self.min_velocity, # x4 2
        #     -np.pi, # x5 3
        #     self.max_steering_angle, 4
        #     self.max_velocity, 5
        #     np.pi, 6
        #     -self.max_steering_vel, # u1 7
        #     -self.max_acceleration, # u2 8
        #     self.max_steering_vel, 9 
        #     self.max_acceleration 10
        # ])




error_v_sum = 0
error_phi_sum = 0

last_error_v = 0
last_error_phi = 0
dt = 0.001
def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    global error_v_sum
    global error_phi_sum
    global last_error_v
    global last_error_phi
    # [steer angle, velocity]
    assert(desired.shape == (2,))

    # differnce signals 
    error_v = desired[1] - state[3]
    error_phi = desired[0] - state[2]

    # accel
    # P  
    K_p_accel = 2
    desired_accel_p = K_p_accel * error_v 
    # I
    K_i_accel = 0
    error_v_sum += dt * error_v
    desired_accel_i = K_i_accel * error_v_sum
    # D
    K_d_accel = 0
    desired_accel_d = K_d_accel * (error_v - last_error_v) / dt
    last_error_v = error_v


    desired_accel = np.clip(
        desired_accel_p + desired_accel_i + desired_accel_d, 
        parameters[8], 
        parameters[10])
    # steering  
    # P
    K_p_steering = 1000
    desired_steering_p = K_p_steering * error_phi

    # I
    K_i_steering = 1
    error_phi_sum += dt * error_phi
    desired_steering_i = K_i_steering * error_phi_sum


    # D
    K_d_steering = 0.1
    desired_steering_d = K_d_steering * (error_phi - last_error_phi) / dt
    last_error_phi = error_phi

    desired_steering = np.clip(
        desired_steering_p + desired_steering_i + desired_steering_d,
        parameters[7],
        parameters[9]
    )

    return np.array([desired_steering, desired_accel]).T


# global state
i = 0 # current index
# constants
distance_threshold = 20

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    global i
    # velocity

    # compute desired angle
    sx = state[0]
    sy = state[1]
    currentHeading = state[4] 
    rx, ry = racetrack.centerline[i]
    while (sx - rx) ** 2 + (sy - ry) ** 2 <= distance_threshold ** 2:
        i = (i + 1) % len(racetrack.centerline) 
        rx, ry = racetrack.centerline[i]

    plt.plot([rx], [ry], 'o')

    dx = rx - sx
    dy = ry - sy

    # pure pursuit

    lwb = parameters[0]
    alpha = np.arctan2((ry - sy), (rx - sx)) - currentHeading

    distance_to_target = np.sqrt(dx**2 + dy**2)
    lookahead = max(lwb, distance_to_target)
    
    desired_angle = np.arctan2(2 * lwb * np.sin(alpha), lookahead)
    desired_angle = np.clip(desired_angle, parameters[1], parameters[4])

    desired_velocity =  0.3 * parameters[5] * max(1.0 - 3 * np.abs(desired_angle) / parameters[4], 0.1)
    return np.array([desired_angle, desired_velocity]).T