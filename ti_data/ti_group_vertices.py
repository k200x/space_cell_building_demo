import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import sort

from ti_data.ti_model import Truncated_Icosahedron
from ti_util.geometry_util import find_truncated_icosahedron_center_from_surface_center_points
from ti_util.renderer_util import RESET, RED

phi = (1 + np.sqrt(5)) / 2


def truncated_icosahedron_surface_center_vertices():
    def truncated_icosahedron_vertices():
        def permutate(A):
            At = np.array(A).T
            Ax = At.flatten()
            Ay = At[[1, 2, 0]].flatten()
            Az = At[[2, 0, 1]].flatten()
            return np.array([Ax, Ay, Az]).T

        pm = [1.0, -1.0]
        A = [[0, y * 1.0, z * 3 * phi] for y in pm for z in pm]
        B = [[x * 1, y * (2 + phi), z * 2 * phi] for x in pm for y in pm for z in pm]
        C = [[x * phi, y * 2, z * (2 * phi + 1)] for x in pm for y in pm for z in pm]
        A = permutate(A)
        B = permutate(B)
        C = permutate(C)
        vs = np.concatenate((A, B, C), axis=0)
        norm = np.linalg.norm(vs[0])
        vs = vs / norm
        return vs

    vs = truncated_icosahedron_vertices()
    # vs_center = find_truncated_icosahedron_center(vs)

    adj = [[] for _ in range(60)]
    edge_length = 0.40354821233519766
    tolerance = 0.001
    for i in range(60):
        for j in range(i + 1, 60):
            d = np.linalg.norm(vs[i] - vs[j])
            if abs(d - edge_length) < tolerance:
                adj[i].append(j)
                adj[j].append(i)

    ordered_neighbors = [None] * 60
    for v in range(60):
        normal = vs[v]
        neighbors = adj[v]
        if normal[0] != 0:
            u = np.array([0, -normal[2], normal[1]])
        elif normal[1] != 0:
            u = np.array([normal[2], 0, -normal[0]])
        else:
            u = np.array([normal[1], -normal[0], 0])
        u = u / np.linalg.norm(u)
        w = np.cross(normal, u)
        angle_tuples = []
        for n in neighbors:
            proj = vs[n] - np.dot(vs[n], normal) * normal
            proj_norm = np.linalg.norm(proj)
            if proj_norm > 0:
                proj /= proj_norm
            x = np.dot(proj, u)
            y = np.dot(proj, w)
            angle = np.arctan2(y, x)
            angle_tuples.append((angle, n))
        angle_tuples.sort(key=lambda x: x[0])
        ordered = [t[1] for t in angle_tuples]
        n0, n1, n2 = ordered
        a = vs[n1] - vs[n0]
        b = vs[n2] - vs[n0]
        cross = np.cross(a, b)
        if np.dot(cross, normal) < 0:
            ordered = ordered[::-1]
        ordered_neighbors[v] = ordered

    potential_faces = set()
    for start in range(60):
        ord_s = ordered_neighbors[start]
        for k in range(3):
            next1 = ord_s[k]
            for direction in [1, -1]:
                face = [start]
                previous = start
                current = next1
                closed = False
                count = 0
                while True:
                    face.append(current)
                    if current == start:
                        closed = True
                        break
                    ord_c = ordered_neighbors[current]
                    pos = ord_c.index(previous)
                    next_current = ord_c[(pos + direction) % 3]
                    previous = current
                    current = next_current
                    count += 1
                    if count > 10:
                        break
                if closed:
                    size = len(face) - 1
                    if size in [5, 6]:
                        face = face[:-1]
                        normalized = tuple(sorted(face))
                        potential_faces.add(normalized)

    surface_center_pts = []
    for f in potential_faces:
        f_list = list(f)
        surface_center_pt = np.mean(vs[f_list], axis=0)
        surface_center_pts.append(surface_center_pt)

    return np.array(surface_center_pts)


def rotation_matrix(axis, theta):
    """
    Generate a rotation matrix for a given axis and angle.
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])


def generate_multi_truncated_icosahedrons(
    obj_num: int,
    ax: Axes3D,
    ball_size_scale: float,
    objs_distance_limit: float,
    obj_surface_num: int,
    space_destination_direction: np.ndarray,
    ti_objs: list
) -> list:
    """
    Generate obj_num truncated icosahedrons connected surface-to-surface, like LEGO bricks.
    """

    base_obj_center = np.linalg.norm([0.0, 0.0, 0.0])

    # Get base icosahedron surface centers
    base_surface_centers = truncated_icosahedron_surface_center_vertices()
    # from original 1.0 axis to reality 1.0 -> 5.0m radiu, and approximately 10.0 ball diameter axis
    base_surface_centers = base_surface_centers * 5.0
    base_ball_radius = [np.linalg.norm(c - base_obj_center) for c in base_surface_centers]
    base_ball_radius = sort(base_ball_radius)

    print("basic center:", base_obj_center)
    print("basic surface center radius:", base_ball_radius)

    # todo (Used for reality)
    # base_obj_center = find_truncated_icosahedron_center_from_surface_center_points(base_surface_centers)

    # average radius meter from polygon and hexagon surface to center
    average_ball_radius = (min(base_ball_radius) + max(base_ball_radius)) / 2  # >4.0m
    print("average ball radius", average_ball_radius)

    print("basic radius", [np.linalg.norm(base_obj_center - s) for s in base_surface_centers])
    scale = ball_size_scale  # Scale factor for icosahedron size
    base_surface_centers *= scale

    # Initialize matrix
    translations = [np.array([0.0, 0.0, 0.0])]  # First icosahedron at origin
    rotations = [np.eye(3)]  # No rotation for first icosahedron

    ti_objs.append(
        Truncated_Icosahedron(
            obj_center=find_truncated_icosahedron_center_from_surface_center_points(
                base_surface_centers @ rotations[0] + translations[0] + space_destination_direction
            ),
            vertices=None,
            surface_center_vertices=(
                base_surface_centers @ rotations[0] + translations[0] + space_destination_direction
            )
        )
    )

    connected_to = []  # Track which icosahedron each is connected to
    connected_faces = []

    rng = np.random.default_rng()

    def select_random_face(surface_center_vertices):
        face_idx_curr = rng.integers(0, obj_surface_num)
        face_curr = surface_center_vertices[face_idx_curr]
        return face_curr

    def get_final_direction(prev_obj_center, select_surface):
        next_obj_face_direction = select_surface - np.array([0.0, 0.0, 0.0])
        prev_obj_to_next_obj_vector = next_obj_face_direction * 2.0
        next_obj_final_direction = prev_obj_center + prev_obj_to_next_obj_vector

        print(
            "check 2 obj centers and distance: ",
            "c1", prev_obj_center,
            "c2", next_obj_final_direction,
            "distance", np.linalg.norm(prev_obj_center - next_obj_final_direction)
        )
        return next_obj_final_direction

    # Place remaining obj_num truncated icosahedrons
    for i in range(0, obj_num):
        print("\nPlace a new ball No.", i + 1)
        """
        Step 1: Select expanded ball and surface
        """
        # ------- choose a ball randomly to expand new ball
        connect_to = rng.integers(0, i + 1)  # Connect to one of 0 to i-1
        prev_obj = ti_objs[connect_to]

        # ------- select a random face from previous icosahedron
        face_idx_prev = rng.integers(0, obj_surface_num)

        while any(fc[0] == connect_to and fc[1] == face_idx_prev for fc in connected_faces):
            face_idx_prev = rng.integers(0, obj_surface_num)  # Ensure unused face

        # ------- select a random face from current icosahedron
        face_curr = select_random_face(base_surface_centers)

        # ------- vector from prev ti_obj to new ti_obj
        next_obj_final_direction = get_final_direction(prev_obj.obj_center, face_curr)

        # ------- check colliding with remain objects
        passed_test = True
        while passed_test:
            passed_test = False
            count = 0  # avoid infinite loop, give 10 times
            for obj in ti_objs:
                old_new_obj_distance = np.linalg.norm(next_obj_final_direction - obj.obj_center)
                print("colliding distance check", old_new_obj_distance)
                if old_new_obj_distance < objs_distance_limit:  # collide
                    print(f"{RED}Oops, select another surface!{RESET}")
                    face_curr = select_random_face(base_surface_centers)
                    next_obj_final_direction = get_final_direction(prev_obj.obj_center, face_curr)
                    passed_test = True
            count += 1
            if count >= 10:
                break

        """
        Step 2: Translate so surfaces coincide
        """
        # todo (no rotation this version)
        # truncated_icosahedron_rotation()
        R = np.eye(3)
        # rotated_face_curr = R @ face_curr

        # translation to make surfaces coincident
        t = next_obj_final_direction

        # Apply transformation
        # transformed_centers = base_surface_centers @ R.T + t
        transformed_centers = base_surface_centers + t
        print(
            "transformed center",
            find_truncated_icosahedron_center_from_surface_center_points(transformed_centers)
        )
        print("prev_obj center", prev_obj.obj_center)
        print(
            "transformed distance", np.linalg.norm(
                prev_obj.obj_center - find_truncated_icosahedron_center_from_surface_center_points(transformed_centers)
            )
        )

        # todo (no rotation this version)
        # translations.append(t)
        # rotations.append(R)

        connected_to.append(connect_to)
        connected_faces.append((connect_to, face_idx_prev))

        ti_obj = Truncated_Icosahedron(
            obj_center=find_truncated_icosahedron_center_from_surface_center_points(
                transformed_centers
            ),
            vertices=None,
            surface_center_vertices=(
                transformed_centers
            )
        )

        # draw it
        ti_obj.draw_obj(ax)

        ti_objs.append(ti_obj)

    print("Totally ti_objs num", len(ti_objs) - 1)

    # remote the virtual root ball
    return ti_objs[1:]
