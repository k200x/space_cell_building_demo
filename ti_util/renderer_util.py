import numpy as np
from scipy.spatial import ConvexHull

# ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def draw_convex_hull(vs_data, ax):
    """
    draw truncated icosahedron
    """
    print("draw")
    hull = ConvexHull(vs_data)
    for f in hull.simplices:
        tri = vs_data[f]
        tri = np.vstack([tri, tri[0]])
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], c='orange', alpha=0.3)
