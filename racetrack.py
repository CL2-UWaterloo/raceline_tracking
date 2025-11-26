import numpy as np

import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.axes as axes

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

        # Precompute curvature estimates along the centerline.
        # For each centerline point i we compute a curvature estimate
        # based on the change in heading over the next `curvature_span` points.
        # Then we compute the maximum curvature over a lookahead window
        # (`max_lookahead`) for use in speed planning.
        try:
            n = self.centerline.shape[0]
            curvature_span = 5
            max_lookahead = 30
            eps = 1e-9

            # helper headings between consecutive points
            headings = np.zeros(n)
            for i in range(n):
                p = self.centerline[i]
                q = self.centerline[(i + 1) % n]
                headings[i] = np.arctan2(q[1] - p[1], q[0] - p[0])

            kappas = np.zeros(n)
            for i in range(n):
                start_idx = i
                end_idx = (i + curvature_span) % n

                theta_start = headings[start_idx]
                theta_end = headings[(end_idx - 1) % n]
                # wrap difference
                dtheta = (theta_end - theta_start + np.pi) % (2 * np.pi) - np.pi

                # arc-length across the span
                ds = 0.0
                for j in range(curvature_span):
                    a = self.centerline[(i + j) % n]
                    b = self.centerline[(i + j + 1) % n]
                    ds += np.linalg.norm(b - a)

                kappas[i] = abs(dtheta) / (ds + eps)

            # For each point compute the maximum curvature over the next max_lookahead points
            kappa_max = np.zeros(n)
            for i in range(n):
                vals = []
                for j in range(max_lookahead):
                    vals.append(kappas[(i + j) % n])
                kappa_max[i] = max(vals) if len(vals) > 0 else 0.0

            self.kappa = kappas
            self.kappa_max_lookahead = kappa_max
            self._curvature_span = curvature_span
            self._kappa_lookahead = max_lookahead
            # Precompute a desired speed map based on upcoming curvature.
            # Use a percentile of the upcoming kappas to be robust to single-point spikes,
            # convert curvature -> speed via lateral-accel constraint: v = sqrt(a_lat / kappa),
            # then smooth the map with a small moving average.
            try:
                pct = 95
                a_lat_max = 20
                safety = 0.5
                v_max_default = 100.0
                v_min_default = 5.0
                smooth_window = 5

                # For each centerline index, compute the percentile curvature over the
                # next `max_lookahead` points and convert to speed.
                desired_speed = np.zeros(n)
                for i in range(n):
                    window = [kappas[(i + j) % n] for j in range(max_lookahead)]
                    if len(window) == 0:
                        kappa_stat = 0.0
                    else:
                        kappa_stat = float(np.percentile(window, pct))

                    if kappa_stat <= 1e-9:
                        v = v_max_default
                    else:
                        v = np.sqrt((a_lat_max * safety) / kappa_stat)

                    desired_speed[i] = float(np.clip(v, v_min_default, v_max_default))

                # smooth desired speed with moving average to avoid rapid local jumps
                kernel = np.ones(smooth_window) / smooth_window
                # pad for circular convolution
                pad = smooth_window // 2
                padded = np.concatenate([desired_speed[-pad:], desired_speed, desired_speed[:pad]])
                smoothed = np.convolve(padded, kernel, mode='valid')
                # smoothed length should equal n
                self.desired_speed_map = smoothed[:n]
            except Exception:
                self.desired_speed_map = np.ones(n) * 20.0
            # Debug prints: curvature statistics and a small sample
            try:
                print(f"[RaceTrack] curvature_span={curvature_span}, kappa_lookahead={max_lookahead}")
                print(f"[RaceTrack] kappa: min={self.kappa.min():.6f}, max={self.kappa.max():.6f}, mean={self.kappa.mean():.6f}")
                print(f"[RaceTrack] kappa_max_lookahead: min={self.kappa_max_lookahead.min():.6f}, max={self.kappa_max_lookahead.max():.6f}, mean={self.kappa_max_lookahead.mean():.6f}")
                # show first 10 curvature values and first 10 lookahead maxima
                np.set_printoptions(precision=6, suppress=True)
                print("[RaceTrack] kappa sample:", self.kappa[:10])
                print("[RaceTrack] kappa_max_lookahead sample:", self.kappa_max_lookahead[:10])
                # show indices of largest curvatures
                top_idx = np.argsort(self.kappa_max_lookahead)[-10:][::-1]
                print("[RaceTrack] top kappa_max_lookahead indices:", top_idx)
                print("[RaceTrack] top kappa_max_lookahead values:", self.kappa_max_lookahead[top_idx])
            except Exception:
                pass
        except Exception:
            # If something goes wrong (e.g., too few points), provide safe defaults
            self.kappa = np.zeros(self.centerline.shape[0])
            self.kappa_max_lookahead = np.zeros(self.centerline.shape[0])
            self._curvature_span = 0
            self._kappa_lookahead = 0

        

    def plot_track(self, axis : axes.Axes):
        axis.add_patch(self.mpl_centerline_patch)
        axis.add_patch(self.mpl_right_track_limit_patch)
        axis.add_patch(self.mpl_left_track_limit_patch)