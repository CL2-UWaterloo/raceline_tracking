import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

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
    

    # a
    desired_accel = 0
    if (d_v > 0):
        desired_accel = parameters[10]
    elif (d_v < 0):
        desired_accel = parameters[8]
    
    # steering 
    desired_steering = 0
 
    if (d_phi > 0):
        desired_steering = parameters[4]
    elif (d_phi < 0):
        desired_steering = parameters[1]

    desired_steering *= abs(d_phi) / parameters[6]    

    return np.array([desired_steering, desired_accel]).T


# global state
i = 0 # current index
# constants
distance_threshold = 5

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    global i
    # velocity
    desired_velocity = 0.1 * parameters[5] 

    # compute desired angle
    sx = state[0]
    sy = state[1]
    currentHeading = state[4] 
    rx, ry = racetrack.centerline[i]
    if (sx - rx) ** 2 + (sy - ry) ** 2 <= distance_threshold ** 2:
        i = (i + 1) % len(racetrack.centerline) 
        rx, ry = racetrack.centerline[i]
    

    lwb = parameters[0]

    coeff = lwb / desired_velocity
    nextHeading = None
    if rx - sx == 0:
        if ry - sy > 0:
            nextHeading = parameters[6] / 2
        else:
            nextHeading = parameters[3] / 2
    else:
        nextHeading = np.arctan2((ry - sy), (rx - sx)) 
    desired_angle = coeff * (nextHeading - currentHeading)
        
    return np.array([desired_angle, desired_velocity]).T