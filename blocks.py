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
        # TODO: Add reference to racecar
        self.base_velocity = 50.0  # m/s
        self.lookahead_distance = 20  # index points
        self.curvature_slowdown_threshold = (
            0.01  # curvature threshold for speed slowdown
        )
        self.curvature_speedup_threshold = (
            0.005  # curvature threshold for speed speedup
        )
        # self.step_size = 1

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

        return np.clip(reference_velocity, -10, 100)
