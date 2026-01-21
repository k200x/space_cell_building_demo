import numpy as np

from ti_util.renderer_util import draw_convex_hull


# todo (It could use dataclass in more than 3.6 version python for high performance.)
class Truncated_Icosahedron:
    def __init__(
        self,
        obj_center: np.ndarray = None,
        vertices: np.ndarray = None,
        surface_center_vertices: np.ndarray = None,
    ):
        self.obj_center = obj_center
        self.vertices = vertices
        self.transformed_vertices = None
        self.surface_center_vertices = surface_center_vertices
        self.transformed_surface_center_vertices = None

    def draw_obj(self, ax):
        draw_convex_hull(self.surface_center_vertices, ax)
