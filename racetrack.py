import numpy as np

import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.axes as axes

def wrap_to_pi(angle : float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi    

def menger_curvature(p1, p2, p3):
        # Calculate side lengths
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        # Calculate semi-perimeter
        s = (a + b + c) / 2

        # Calculate area using Heron's formula
        try:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        except ValueError: # Occurs if s*(s-a)*(s-b)*(s-c) is slightly negative due to precision
            area = 0.0

        # Calculate curvature
        if a * b * c == 0 or area == 0: # Handle collinear points or degenerate triangles
            return 0.0
        else:
            return (4 * area) / (a * b * c)

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
        curvature_v_max = np.zeros(n)
        
        max_v = 100.0          # Maximum speed on straights (m/s)
        friction_limit = 20.0  # Determines speed around curves
        max_accel = 20.0
        
        # compute max speed purely based on curvature
        for i in range(n):

            p1 = self.centerline[(i - 1) % n]
            p2 = self.centerline[i]
            p3 = self.centerline[(i + 1) % n]
            
            # Calculate radius of the circle passing through p1, p2, p3 using Menger's curvature formula
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Angle between segments
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = np.abs(wrap_to_pi(angle))
            
            # If angle is very small, we are basically in a straight line so go at max speed
            if angle < 1e-4:
                curvature_v_max[i] = max_v
            else:
                curvature = menger_curvature(p1, p2, p3)
                radius = 1 / (curvature + 1e-6)
                curvature_v_max[i] = np.sqrt(friction_limit * radius)

        self.desired_speed = np.clip(curvature_v_max, 0, max_v)
        # make sure that for each desired velocity, we can actually hit it based on our max acceleration
        for i in range(n - 1, -1, -1):
            next_idx = (i + 1) % n
            d = np.linalg.norm(self.centerline[next_idx] - self.centerline[i])

            # compute max speed using v_f^2 = v_i^2 + 2 * a * d and taking the min with curvature-based speed
            allowed_v = np.sqrt(self.desired_speed[next_idx]**2 + 2 * max_accel * d)
            self.desired_speed[i] = min(self.desired_speed[i], allowed_v)

    def plot_track(self, axis : axes.Axes):
        axis.add_patch(self.mpl_centerline_patch)
        axis.add_patch(self.mpl_right_track_limit_patch)
        axis.add_patch(self.mpl_left_track_limit_patch)