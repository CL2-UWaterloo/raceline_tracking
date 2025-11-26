import numpy as np

import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.axes as axes

def wrap_to_pi(angle : float) -> float:
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi    

class RaceTrack:

    def __init__(self, filepath : str):
        data = np.loadtxt(filepath, comments="#", delimiter=",")
        self.centerline = data[:, 0:2]
        self.centerline = np.vstack((self.centerline[-1], self.centerline, self.centerline[0]))

        centerline_gradient = np.gradient(self.centerline, axis=0)
        # Unfortunate Warning Print: https://github.com/numpy/numpy/issues/26620
        centerline_cross = np.cross(centerline_gradient, np.array([0.0, 0.0, 1.0]))
        centerline_norm = centerline_cross*\
            np.divide(1.0, np.linalg.norm(centerline_cross, axis=1))[:, None]

        centerline_norm = np.delete(centerline_norm, 0, axis=0)
        centerline_norm = np.delete(centerline_norm, -1, axis=0)

        self.centerline = np.delete(self.centerline, 0, axis=0)
        self.centerline = np.delete(self.centerline, -1, axis=0)

        # Compute track left and right boundaries
        self.right_boundary = self.centerline[:, :2] + centerline_norm[:, :2] * np.expand_dims(data[:, 2], axis=1)
        self.left_boundary = self.centerline[:, :2] - centerline_norm[:, :2]*np.expand_dims(data[:, 3], axis=1)

        # Compute initial position and heading
        self.initial_state = np.array([
            self.centerline[0, 0],
            self.centerline[0, 1],
            0.0, 0.0,
            np.arctan2(
                self.centerline[1, 1] - self.centerline[0, 1], 
                self.centerline[1, 0] - self.centerline[0, 0]
            )
        ])

        # Matplotlib Plots
        self.code = np.empty(self.centerline.shape[0], dtype=np.uint8)
        self.code.fill(path.Path.LINETO)
        self.code[0] = path.Path.MOVETO
        self.code[-1] = path.Path.CLOSEPOLY

        self.mpl_centerline = path.Path(self.centerline, self.code)
        self.mpl_right_track_limit = path.Path(self.right_boundary, self.code)
        self.mpl_left_track_limit = path.Path(self.left_boundary, self.code)

        self.mpl_centerline_patch = patches.PathPatch(self.mpl_centerline, linestyle="-", fill=False, lw=0.3)
        self.mpl_right_track_limit_patch = patches.PathPatch(self.mpl_right_track_limit, linestyle="--", fill=False, lw=0.2)
        self.mpl_left_track_limit_patch = patches.PathPatch(self.mpl_left_track_limit, linestyle="--", fill=False, lw=0.2)

        n = self.centerline.shape[0]
        self.desired_speed = np.zeros(n)
        curvature_v_max = np.zeros(n)
        
        max_v = 100.0          # Maximum speed on straights (m/s)
        friction_limit = 20.0  # Determines speed around curves
        max_accel = 20.0
        
        # compute "ideal" speed purely based on curvature
        for i in range(n):

            p1 = self.centerline[(i - 1) % n]
            p2 = self.centerline[i]
            p3 = self.centerline[(i + 1) % n]
            
            # Calculate radius of the circle passing through p1, p2, p3
            # Using Menger curvature formula or simple geometric approximation
            # Approximate geometric curvature: k = 2 * sin(ang) / |p1-p3|
            
            v1 = p2 - p1
            v2 = p3 - p2
            dist_segment = np.linalg.norm(v1) # Approximate distance
            
            # Angle between segments
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = np.abs(wrap_to_pi(angle))
            
            # If angle is very small, we are in a straight line
            if angle < 1e-4:
                curvature_v_max[i] = max_v
            else:
                # R ~ distance / angle (small angle approximation)
                # v^2 / R = a_lat  =>  v = sqrt(R * a_lat)
                radius = dist_segment / (angle + 1e-6)
                curvature_v_max[i] = np.sqrt(friction_limit * radius)

        profile = np.clip(curvature_v_max, 0, max_v)
        # Backpropragation to make sure that for each desired velocity, we can actually hit it based on our max acceleration
        for i in range(n - 1, -1, -1):
            # The next point (conceptually ahead of us)
            next_idx = (i + 1) % n
            
            dist = np.linalg.norm(self.centerline[next_idx] - self.centerline[i])
            
            # Allowed entry speed at 'i' to reach speed at 'i+1' using max braking
            # v_i^2 = v_{i+1}^2 + 2 * a * d
            allowed_v = np.sqrt(profile[next_idx]**2 + 2 * max_accel * dist)
            
            # Take the minimum of the physical corner limit and the braking limit
            profile[i] = min(profile[i], allowed_v)
        # Same thing but in the forward directionn
        for i in range(n):
            prev_idx = (i - 1) % n
            dist = np.linalg.norm(self.centerline[i] - self.centerline[prev_idx])
            
            allowed_v = np.sqrt(profile[prev_idx]**2 + 2 * max_accel * dist)
            profile[i] = min(profile[i], allowed_v)

        self.desired_speed = profile

    def plot_track(self, axis : axes.Axes):
        axis.add_patch(self.mpl_centerline_patch)
        axis.add_patch(self.mpl_right_track_limit_patch)
        axis.add_patch(self.mpl_left_track_limit_patch)