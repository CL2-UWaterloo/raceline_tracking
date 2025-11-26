import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack

class PIDController:
    """
    Discrete-Time PID Controller with Anti-Windup and Derivative Filtering.
    
    Implements:
    u[k] = P[k] + I[k] + D[k]
    
    P[k] = Kp * e[k]
    I[k] = I[k-1] + Ki * Ts * (e[k] + e[k-1]) / 2  (Tustin / Trapezoidal)
    D[k] = Kd * (e[k] - e[k-1]) / Ts               (Backward Difference)
    
    Anti-Windup: Clamps the integrator component if output saturates.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, ts: float, min_out: float, max_out: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ts = ts
        self.min_out = min_out
        self.max_out = max_out
        
        # State
        self.prev_error = 0.0
        self.integral = 0.0
        self.reset()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error: float) -> float:
        # Proportional
        p_term = self.kp * error
        
        # Integral (Tustin approximation)
        # I[k] = I[k-1] + (Ki * Ts / 2) * (e[k] + e[k-1])
        # We compute a candidate integral first
        delta_integral = (self.ki * self.ts / 2.0) * (error + self.prev_error)
        candidate_integral = self.integral + delta_integral
        
        # Derivative (Backward Euler)
        d_term = self.kd * (error - self.prev_error) / self.ts
        
        # Compute raw output
        raw_output = p_term + candidate_integral + d_term
        
        # Output Saturation & Anti-Windup
        # If we are saturating, we do not update the integral term to accumulate more error (Clamping)
        output = np.clip(raw_output, self.min_out, self.max_out)
        
        # Simple Anti-Windup: Only update integral if we are not saturated 
        # or if the error attempts to bring us back from saturation.
        # Check if saturation occurred
        if raw_output != output:
             # Sign mismatch or magnitude issue. 
             # If driving further into saturation, don't integrate.
             # If driving out of saturation, do integrate.
             if np.sign(error) == np.sign(raw_output):
                 # Driving further into saturation -> clamp integral (keep previous)
                 pass 
             else:
                 # Driving out -> allow integration
                 self.integral = candidate_integral
        else:
            self.integral = candidate_integral
            
        self.prev_error = error
        
        return output

class VehicleController:
    """
    Hierarchical Controller:
    1. Outer Loop (Path Tracking): Pure Pursuit -> Desired Steering Angle, Velocity
    2. Inner Loop (Dynamics): PID Controllers -> Steering Rate, Acceleration
    """
    
    def __init__(self, parameters: ArrayLike, raceline_data: ArrayLike = None):
        self.parameters = parameters
        
        # Extract Vehicle limits for PID clamping
        # parameters: [L, -max_steer, min_vel, -pi, max_steer, max_vel, pi, -max_steer_rate, -max_accel, max_steer_rate, max_accel]
        self.L = parameters[0]
        self.max_steer_angle = parameters[4]
        self.max_velocity = parameters[5]
        self.max_steer_rate = parameters[9]
        self.max_accel = parameters[10]
        self.min_accel = parameters[8] # Usually negative max accel
        
        self.ts = 0.1 # 100ms sample time (from RaceCar time_step)

        # Path to track (Raceline or Centerline)
        self.path = raceline_data
        
        # Track the last known good index to prevent backward searching
        self.last_closest_idx = 0
        
        # Precompute path curvature for velocity planning
        self.path_curvature = None
        if self.path is not None:
            self._precompute_curvature()
        
        # Tunable Control Parameters
        # Velocity-adaptive lookahead: base + gain * velocity
        self.lookahead_base = 5.0      # Minimum lookahead distance (m)
        self.lookahead_gain = 0.7      # Balanced: tight in corners, smooth on straights
        
        # Physics-based velocity limits
        self.max_straight_velocity = 90.0   # Top speed on straights (m/s)
        self.min_velocity = 20.0            # Absolute minimum (safety)
        self.curvature_lookahead = 45       # Balance: not too early, not too late
        
        # Lateral grip limit - max lateral acceleration before losing traction
        # v_max = sqrt(a_lat_max / curvature)
        # Balanced settings: safe in corners, fast on straights
        self.max_lateral_accel = 14.0       # m/s^2 (~1.4g) - balanced
        self.grip_margin = 0.82             # Use 82% of grip - good balance
        
        # PID Tunings
        # Steering Rate PID - reduced gains to prevent oscillation
        kp_steer = 5.0    # Reduced from 8.0 - less aggressive
        ki_steer = 0.5    # Reduced from 2.0 - less integral windup
        kd_steer = 0.2    # Increased slightly for damping
        
        # Velocity PID - more aggressive for faster acceleration
        kp_vel = 3.0      # Increased for quicker response on straights
        ki_vel = 0.5      # Increased to eliminate steady-state error
        kd_vel = 0.1      # Added for smoother transitions
        
        self.steer_pid = PIDController(
            kp_steer, ki_steer, kd_steer, self.ts, 
            -self.max_steer_rate, self.max_steer_rate
        )
        
        self.velocity_pid = PIDController(
            kp_vel, ki_vel, kd_vel, self.ts,
            self.min_accel, self.max_accel
        )

    def set_path(self, path: ArrayLike):
        self.path = path
        if self.path is not None:
            self._precompute_curvature()
    
    def _precompute_curvature(self):
        """
        Precompute curvature at each point on the path.
        Curvature κ = |dT/ds| where T is the unit tangent vector.
        For discrete points: κ ≈ |Δθ| / |Δs|
        """
        n = len(self.path)
        self.path_curvature = np.zeros(n)
        
        for i in range(n):
            # Get three consecutive points
            p0 = self.path[(i - 1) % n]
            p1 = self.path[i]
            p2 = self.path[(i + 1) % n]
            
            # Vectors
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Lengths
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 < 1e-6 or len2 < 1e-6:
                self.path_curvature[i] = 0.0
                continue
            
            # Angle change (using cross product for signed curvature)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = np.dot(v1, v2)
            angle_change = np.arctan2(cross, dot)
            
            # Arc length approximation
            ds = (len1 + len2) / 2.0
            
            # Curvature magnitude
            self.path_curvature[i] = abs(angle_change) / ds
    
    def _get_target_velocity(self, closest_idx):
        """
        Calculate target velocity based on physics - the maximum speed that 
        keeps lateral acceleration within grip limits.
        
        Physics: a_lateral = v^2 * curvature
        Therefore: v_max = sqrt(a_lat_max / curvature)
        """
        if self.path_curvature is None:
            return self.min_velocity
        
        n = len(self.path)
        
        # Look ahead and find the maximum curvature in the upcoming section
        max_curvature = 0.0
        for i in range(self.curvature_lookahead):
            idx = (closest_idx + i) % n
            max_curvature = max(max_curvature, self.path_curvature[idx])
        
        # Physics-based maximum velocity
        # v_max = sqrt(a_lateral_max / curvature)
        if max_curvature < 1e-6:
            # Essentially straight - go max speed
            return self.max_straight_velocity
        
        # Calculate physics-limited speed with safety margin
        v_physics = np.sqrt(self.max_lateral_accel / max_curvature) * self.grip_margin
        
        # Clamp to vehicle limits
        target_vel = np.clip(v_physics, self.min_velocity, self.max_straight_velocity)
        
        return target_vel

    def compute_control(self, state: ArrayLike) -> ArrayLike:
        """
        Computes control inputs [steering_rate, acceleration]
        state: [x, y, delta, v, psi]
        """
        # 1. Outer Loop: Pure Pursuit
        desired_steer, desired_vel = self._pure_pursuit(state)
        
        # 2. Inner Loop: PID Control
        current_steer = state[2]
        current_vel = state[3]
        
        steer_error = desired_steer - current_steer
        vel_error = desired_vel - current_vel
        
        steering_rate = self.steer_pid.update(steer_error)
        acceleration = self.velocity_pid.update(vel_error)
        
        return np.array([steering_rate, acceleration])

    def _pure_pursuit(self, state):
        if self.path is None:
            return 0.0, 0.0
            
        x, y, delta, v, psi = state
        n_points = len(self.path)
        
        # Velocity-adaptive lookahead distance
        current_speed = max(abs(v), 1.0)
        lookahead_dist = self.lookahead_base + self.lookahead_gain * current_speed
        
        # Find closest point - use GLOBAL search (more robust)
        car_pos = np.array([x, y])
        dists = np.linalg.norm(self.path - car_pos, axis=1)
        closest_idx = np.argmin(dists)
        
        # Only allow moving forward on the track (prevent backwards jumps)
        # But allow some backward tolerance for when car overshoots
        idx_diff = closest_idx - self.last_closest_idx
        if idx_diff < -10 and idx_diff > -n_points + 100:
            # Big backwards jump (not crossing start/finish) - keep last position
            closest_idx = self.last_closest_idx
        
        self.last_closest_idx = closest_idx
        
        # Find lookahead point - search forward along the path
        lookahead_idx = closest_idx
        accumulated_dist = 0.0
        
        for i in range(1, min(200, n_points // 2)):
            idx = (closest_idx + i) % n_points
            prev_idx = (closest_idx + i - 1) % n_points
            
            # Accumulate distance along the path
            segment_dist = np.linalg.norm(self.path[idx] - self.path[prev_idx])
            accumulated_dist += segment_dist
            
            if accumulated_dist >= lookahead_dist:
                lookahead_idx = idx
                break
        
        # Fallback
        if lookahead_idx == closest_idx:
            lookahead_idx = (closest_idx + 10) % n_points
        
        target = self.path[lookahead_idx]
        
        # Calculate Alpha (angle to target relative to heading)
        dx = target[0] - x
        dy = target[1] - y
        angle_to_target = np.arctan2(dy, dx)
        alpha = angle_to_target - psi
        
        # Normalize alpha to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Pure Pursuit Steering
        lookahead_actual = max(np.linalg.norm([dx, dy]), 0.1)
        desired_steer = np.arctan(2 * self.L * np.sin(alpha) / lookahead_actual)
        
        desired_steer = np.clip(desired_steer, -self.max_steer_angle, self.max_steer_angle)
        
        # Calculate target velocity based on upcoming curvature
        target_vel = self._get_target_velocity(closest_idx)
        
        return desired_steer, target_vel
