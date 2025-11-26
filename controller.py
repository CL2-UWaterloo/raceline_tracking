import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

class PIDController:
    def __init__(self, K_p: float, K_i: float, K_d: float, dt: float = 0.1):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.dt = dt

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error: float) -> float:
        P = self.K_p * error
        
        self.integral += error * self.dt
        I = self.K_i * self.integral
        
        derivative = (error - self.prev_error) / self.dt
        D = self.K_d * derivative

        self.prev_error = error
        
        return P + I + D
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

steering_pid = PIDController(K_p=2, K_i=0.4, K_d=0.1)
velocity_pid = PIDController(K_p=2, K_i=0.4, K_d=0.1)

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # [steer angle, velocity]
    assert(desired.shape == (2,))
    
    # current state
    delta = state[2]
    v = state[3]
    
    # reference state
    r_delta = desired[0]
    r_v = desired[1]

    e_delta = r_delta - delta
    e_v = r_v - v
    
    steering_rate = steering_pid.update(e_delta)
    acceleration = velocity_pid.update(e_v)
    
    return np.array([steering_rate, acceleration]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    # using the x and y coordinates of the car, find the closest point on the racetrack centerline
    
    centerline_distances = np.linalg.norm(racetrack.centerline - state[0:2], axis=1)
    closest_idx = np.argmin(centerline_distances)

    desired_velocity = racetrack.desired_speed_map[closest_idx] # racetrack.desired_speed_map[closest_idx]
    desired_velocity = np.clip(desired_velocity, 5, 100)
    
    lookahead_amt = 4 # int(np.floor(desired_velocity / 10.0) + 1)
    # compute desired heading based on one point ahead on the centerline
    lookahead_idx = (closest_idx + lookahead_amt) % len(racetrack.centerline)
    lookahead_pt = racetrack.centerline[lookahead_idx]
    
    heading = np.arctan2(
        lookahead_pt[1] - state[1],
        lookahead_pt[0] - state[0]
    )
    
    # heading = wrap_to_pi(heading) 
    delta = wrap_to_pi(heading - state[4]) #* 3.6 / np.linalg.norm(lookahead_pt - state[0:2])
    delta = np.clip(delta, -0.9, 0.9)
   
    # Clip outputs to actuator/state bounds defined in RaceCar.parameters

    return np.array([delta, desired_velocity]).T