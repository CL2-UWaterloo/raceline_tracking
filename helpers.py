import numpy as np


def distance_to_centerline(position: np.ndarray, centerline: np.ndarray) -> float:
    """Compute the minimum distance from a position to the racetrack centerline.

    Args:
        position (np.ndarray): The (x, y) position of the vehicle.
        centerline (np.ndarray): The array of (x, y) points defining the racetrack centerline.

    Returns:
        float: The minimum distance to the centerline.
    """
    deltas = centerline - position
    distances = np.linalg.norm(deltas, axis=1)
    return np.min(distances)


def index_of_closest_point(position: np.ndarray, centerline: np.ndarray) -> int:
    """Find the index of the closest point on the centerline to the given position.

    Args:
        position (np.ndarray): The (x, y) position of the vehicle.
        centerline (np.ndarray): The array of (x, y) points defining the racetrack centerline.

    Returns:
        int: The index of the closest point on the centerline.
    """
    deltas = centerline - position
    distances = np.linalg.norm(deltas, axis=1)
    return np.argmin(distances)
