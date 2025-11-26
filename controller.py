import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

class PIDController:
    def __init__(self, K_p: float, K_i: float, K_d: float, dt: float = 1.0):
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

steering_pid = PIDController(K_p=20, K_i=0, K_d=0)
velocity_pid = PIDController(K_p=10, K_i=0, K_d=0)

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
    wheelbase = parameters[0]
    centerline_distances = np.linalg.norm(racetrack.centerline - state[0:2], axis=1)
    closest_idx = np.argmin(centerline_distances)   
    cur_v = float(state[3])
    # interpolate lookahead amount from 1 to 5 based on current velocity with fast ramp-up
    v_min = 0.0
    v_max = 100.0
    lookahead_min = 1
    lookahead_max = 5
    # normalize velocity to [0, 1], apply power function for fast ramp-up, then scale to lookahead range
    normalized_v = np.clip((cur_v - v_min) / (v_max - v_min), 0, 1)
    lookahead_amt = int(np.round(lookahead_min + (lookahead_max - lookahead_min) * (normalized_v ** 0.3)))
    
    # compute desired heading based on one point ahead on the centerline
    lookahead_idx = (closest_idx + lookahead_amt) % len(racetrack.centerline)
    lookahead_pt = racetrack.centerline[lookahead_idx]
    lookahead_vector = lookahead_pt - state[0:2]
    
    heading = np.arctan2(
        lookahead_vector[1], lookahead_vector[0]
    )
    
    # heading = wrap_to_pi(heading) 
    L_d =  np.linalg.norm(lookahead_vector)
    alpha = wrap_to_pi(heading - state[4]) 
    delta = np.arctan(2 * wheelbase * np.sin(alpha) / L_d)
    # Clip outputs bounds defined in RaceCar.parameters
    delta = np.clip(delta, parameters[1], parameters[4])

    desired_velocity = racetrack.desired_speed[closest_idx]
    desired_velocity = float(np.clip(desired_velocity, 5, 100)) # Min velocity of 5 to make sure that the car doesn't stop
    
    return np.array([delta, desired_velocity]).T