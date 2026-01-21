import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

from ti_data.ti_group_vertices import generate_multi_truncated_icosahedrons

# -----------------------------
# Parameters
# -----------------------------
N_PENT = 12  # Polygon Number
N_HEX = 20  # Hexagon Number
OBJ_NUM = 20  # Look at here first!!!
OBJ_SURFACE_NUM = N_PENT + N_HEX

BALL_SIZE_SCALE = 1.0  # keep 1.0 scale time big
total_tiles = 32 * OBJ_NUM
N = total_tiles

# Animation timing parameters
dt = 3  # time step (s)
launch_interval = 2  # each tile starts 1 second later
flight_time = 8  # flight duration of each tile (s)
T_total = launch_interval * (N - 1) + flight_time

# 3D space parameters
AXIS_CUBIC_EDGE_LEN = 40

DISTANCE_FROM_FACTORY_TO_SPACE_DESTINATION = 40.0
DIRECTION_VECTOR = np.array([1.0, 1.0, 1.0])
SPACE_DESTINATION_DIRECTION = (
    DIRECTION_VECTOR / np.linalg.norm(DIRECTION_VECTOR)
    * DISTANCE_FROM_FACTORY_TO_SPACE_DESTINATION
)
OBJS_DISTANCE_LIMIT = 9.14  # truncated icosahedron surface to center min distance

# initialize
ti_objs = []

print("space destination direction", SPACE_DESTINATION_DIRECTION, "\n")

# -----------------------------
# Truncated icosahedron vertices
# -----------------------------
phi = (1 + np.sqrt(5)) / 2

# -----------------------------
# Animation setup
# -----------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, AXIS_CUBIC_EDGE_LEN)
ax.set_ylim(-1, AXIS_CUBIC_EDGE_LEN)
ax.set_zlim(-1, AXIS_CUBIC_EDGE_LEN)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ti_objs = generate_multi_truncated_icosahedrons(
    obj_num=OBJ_NUM,
    ax=ax,
    ball_size_scale=BALL_SIZE_SCALE,
    objs_distance_limit=OBJS_DISTANCE_LIMIT,
    obj_surface_num=OBJ_SURFACE_NUM,
    space_destination_direction=SPACE_DESTINATION_DIRECTION,
    ti_objs=ti_objs
)

vts = [ti.surface_center_vertices for ti in ti_objs]
vertices = np.vstack(vts)

# -----------------------------
# Factory start positions (cube array)
# -----------------------------
grid_size = int(np.ceil(N ** (1 / 3)))
x = np.linspace(-0.5, 0.5, grid_size)
y = np.linspace(-0.5, 0.5, grid_size)
z = np.linspace(-0.5, 0.5, grid_size)
X, Y, Z = np.meshgrid(x, y, z)
p0 = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N]


# -----------------------------
# Assign tiles start -> destination (greedy)
# -----------------------------
def greedy_assign(starts, targets):
    assigned = np.zeros(len(starts), dtype=int)
    remaining = list(range(len(targets)))
    for i, s in enumerate(starts):
        dists = np.linalg.norm(targets[remaining] - s, axis=1)
        idx = np.argmin(dists)
        assigned[i] = remaining[idx]
        remaining.pop(idx)
    return assigned


assign_idx = greedy_assign(p0, vertices)
reversed_assign_idx = assign_idx[::-1]  # build from far to near

pT = vertices[reversed_assign_idx]

# -----------------------------
# Compute Bézier control points
# -----------------------------
control1 = np.zeros_like(p0)
control2 = np.zeros_like(p0)
for i in range(0, N):
    P0 = p0[i]
    P3 = pT[i]
    dir_vec = P3 - P0
    # perpendicular lift vector
    perp = np.cross(dir_vec, [0, 0, 1])
    if np.linalg.norm(perp) < 1e-3:
        perp = np.cross(dir_vec, [0, 1, 0])
    perp /= np.linalg.norm(perp)
    distance = np.linalg.norm(dir_vec)
    lift = 0.5 * distance
    control1[i] = P0 + 0.3 * dir_vec + lift * perp
    control2[i] = P0 + 0.6 * dir_vec + 0.8 * lift * perp


# -----------------------------
# Bézier curve function
# -----------------------------
def bezier(P0, P1, P2, P3, tau):
    return (
        (1 - tau) ** 3 * P0 +
        3 * (1 - tau) ** 2 * tau * P1 +
        3 * (1 - tau) * tau ** 2 * P2 +
        tau ** 3 * P3
    )


# -----------------------------
# Precompute trajectories (each starts every specific second)
# -----------------------------
times = np.arange(0, T_total + dt, dt)
traj_samples = np.zeros((len(times), N, 3))

for ti, t in enumerate(times):
    for i in range(N):
        t_launch = i * launch_interval
        if t < t_launch:
            traj_samples[ti, i] = p0[i]  # not yet launched
        elif t >= t_launch + flight_time:
            traj_samples[ti, i] = pT[i]  # arrived
        else:
            tau = (t - t_launch) / flight_time
            traj_samples[ti, i] = bezier(p0[i], control1[i], control2[i], pT[i], tau)

# Scatter for moving tiles
scat = ax.scatter([], [], [], c='blue', s=50)

# Trajectory trails
lines = [ax.plot([], [], [], c='gray', alpha=0.2)[0] for _ in range(N)]


# -----------------------------
# Update function
# -----------------------------
def update(frame):
    scat._offsets3d = (traj_samples[frame, :, 0],
                       traj_samples[frame, :, 1],
                       traj_samples[frame, :, 2])
    for i, line in enumerate(lines):
        line.set_data(traj_samples[:frame + 1, i, 0], traj_samples[:frame + 1, i, 1])
        line.set_3d_properties(traj_samples[:frame + 1, i, 2])
    return [scat] + lines


anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

# 1.Show in window
plt.show()  # if save mp4, comment this line

# 2.Save as MP4
# writer = FFMpegWriter(fps=20, metadata={'artist': 'Simulation'}, bitrate=1800)
# anim.save("ti_building_20_objs_example.mp4", writer=writer)
