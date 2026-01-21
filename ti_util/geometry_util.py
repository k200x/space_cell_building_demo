import numpy as np


def euclidean_distance(pt1, pt2):
    """
    Calculate Euclidean distance between two points
    """
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


def find_truncated_icosahedron_center(vertices):
    """
    Compute the center of a truncated icosahedron given its 60 vertex positions.
    """
    if vertices.shape != (60, 3):
        raise ValueError("Input must be a (60, 3) NumPy array.")
    return np.mean(vertices, axis=0)


def find_truncated_icosahedron_center_from_surface_center_points(vertices):
    """
    Compute the center of a truncated icosahedron given its 32 surface center vertex positions.
    """
    if vertices.shape != (32, 3):
        raise ValueError("Input must be a (32, 3) NumPy array.")
    return np.mean(vertices, axis=0)


def truncated_icosahedron_rotation():
    """
    todo (should align by pantagon, hexagon surface in future, not just random rotation)
    """
    # Want normal_curr -> -normal_prev
    # dot = np.dot(normal_curr, -normal_prev)
    # if abs(dot - 1) < 1e-8:  # Already aligned
    #     R1 = np.eye(3)
    #     print("a")
    # elif abs(dot + 1) < 1e-8:  # Opposite
    #     axis = np.array([0, 0, 1]) if abs(normal_curr[2]) < 0.9 else np.array([0, 1, 0])
    #     R1 = rotation_matrix(axis, np.pi)
    #     print("b")
    # else:
    #     axis = np.cross(normal_curr, -normal_prev)
    #     axis /= np.linalg.norm(axis)
    #     theta = np.arccos(dot)
    #     R1 = rotation_matrix(axis, theta)
    #     print("c")
    #
    # Apply random in-plane rotation for variety
    # R2 = rotation_matrix(-normal_prev, rng.uniform(0, 2 * np.pi))
    # R = R2 @ R1
    pass
