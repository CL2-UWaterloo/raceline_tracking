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


def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # [steer angle, velocity]
    assert(desired.shape == (2,))

    # differnce signals 
    d_v = desired[1] - state[3]
    d_phi = desired[0] - state[2]
    
    K_p_accel = 2
    desired_accel = K_p_accel * np.clip(d_v * 10, parameters[8], parameters[10])

    
    # steering  
    K_p_steering = 1
    desired_steering = K_p_steering * np.clip(d_phi, parameters[7], parameters[9])

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
    if (sx - rx) ** 2 + (sy - ry) ** 2 <= distance_threshold ** 2:
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

    desired_velocity = 0.2 * parameters[5] * max(1.0 - 2.0 * np.abs(desired_angle) / parameters[4], 0.1)
    print(desired_velocity)
    return np.array([desired_angle, desired_velocity]).T