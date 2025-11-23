"""
This file implements the blocks that are not controllers or plants (S1 and S2).
"""

import numpy as np
import helpers


class S1:
    """
    Using information about the centerline, output a reference velocity.
    """

    def __init__(self):
        self.base_velocity = 50.0  # m/s
        self.lookahead_distance = 20  # index points
        self.curvature_slowdown_threshold = (
            0.01  # curvature threshold for speed slowdown
        )
        self.curvature_speedup_threshold = (
            0.005  # curvature threshold for speed speedup
        )

    def step(self, state: np.ndarray, centerline: np.ndarray) -> float:
        """
        Target velocity should be intelligently be set based on upcoming curvature
        """
        closest_position = helpers.distance_to_centerline(state[0:2], centerline)
        closest_index = helpers.index_of_closest_point(state[0:2], centerline)

        curvatures = []
        for i in range(self.lookahead_distance):
            idx = (closest_index + i) % len(centerline)
            idx_next = (closest_index + i + 1) % len(centerline)
            idx_next_next = (closest_index + i + 2) % len(centerline)

            p1, p2, p3 = (
                centerline[idx],
                centerline[idx_next],
                centerline[idx_next_next],
            )

            # Calculate Menger curvature
            # https://en.wikipedia.org/wiki/Menger_curvature
            area = 0.5 * np.abs(
                p1[0] * (p2[1] - p3[1])
                + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])
            )

            len_a = np.linalg.norm(p1 - p2)
            len_b = np.linalg.norm(p2 - p3)
            len_c = np.linalg.norm(p3 - p1)

            # If area of triangle is negligible
            if np.isclose(area, 0.0):
                curvature = 0.0
            else:
                curvature = (4 * area) / (len_a * len_b * len_c)

            print(f"Curvature at index {idx}: {curvature}")
            curvatures.append(curvature)

        curvature_avg = np.mean(curvatures)
        print(f"Average curvature over lookahead: {curvature_avg}")

        reference_velocity = self.base_velocity
        if curvature_avg > self.curvature_slowdown_threshold:
            # Slow down
            reference_velocity = self.base_velocity * 0.5
        elif curvature_avg < self.curvature_speedup_threshold:
            # Speed up
            reference_velocity = self.base_velocity * 1.2

        print(f"Reference velocity: {reference_velocity}")
        return np.clip(reference_velocity, -10, 100)


class S2:
    def __init__(self):
        self.small_number_threshold = 0.1  # To avoid division by zero

    def step(
        self,
        state: np.ndarray,
        centerline: np.ndarray,
        lookahead_distance: float,
        wheelbase: float,
    ) -> float:
        """
        Implementing a (simplified) pure pursuit algorithm to compute the steering angle.
        """
        x, y = state[0], state[1]
        heading = state[4]
        closest_index = helpers.index_of_closest_point(state[0:2], centerline)

        # Search along path for a lookahead point at sufficient distance
        lookahead_idx = closest_index
        total_dist = 0.0

        for i in range(closest_index, closest_index + len(centerline)):
            idx = i % len(centerline)
            next_idx = (i + 1) % len(centerline)

            point = centerline[idx]
            next_point = centerline[next_idx]

            segment_length = np.linalg.norm(next_point - point)
            total_dist += segment_length

            if total_dist >= lookahead_distance:
                lookahead_idx = next_idx
                break

        lookahead_point = centerline[lookahead_idx]

        # Vector from vehicle to lookahead point
        dx = lookahead_point[0] - x
        dy = lookahead_point[1] - y

        # Transform to vehicle coordinates
        local_x = np.cos(-heading) * dx - np.sin(-heading) * dy
        local_y = np.sin(-heading) * dx + np.cos(-heading) * dy

        # Actual distance to lookahead point
        L_d = np.sqrt(local_x**2 + local_y**2)

        # Prevent division by zero or very small numbers
        if L_d < self.small_number_threshold:
            L_d = self.small_number_threshold

        # Simplified pure pursuit steering:
        curvature = (2 * local_y) / (L_d**2)
        desired_steering_angle = np.arctan(curvature * wheelbase)

        return desired_steering_angle
