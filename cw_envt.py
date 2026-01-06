# === File: cw_envt.py ===
import os
import math
import random
import pybullet as p
import pybullet_data


# ---------------------------
# Arena helpers
# ---------------------------
def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.5

    # Floor
    floor_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness]
    )
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
        rgbaColor=[1, 1, 0, 1],
    )
    floor_id = p.createMultiBody(0, floor_shape, floor_visual, [0, 0, -wall_thickness])

    # Walls (Y walls)
    wall_shape1 = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2]
    )
    wall_visual1 = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
        rgbaColor=[0.7, 0.7, 0.7, 1],
    )
    for y in [arena_size / 2, -arena_size / 2]:
        p.createMultiBody(0, wall_shape1, wall_visual1, [0, y, wall_height / 2])

    # Walls (X walls)
    wall_shape2 = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2]
    )
    wall_visual2 = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
        rgbaColor=[0.7, 0.7, 0.7, 1],
    )
    for x in [arena_size / 2, -arena_size / 2]:
        p.createMultiBody(0, wall_shape2, wall_visual2, [x, 0, wall_height / 2])

    # friction helps climbing
    p.changeDynamics(floor_id, -1, lateralFriction=1.0, rollingFriction=0.02, spinningFriction=0.02)
    return floor_id


# ---------------------------
# Mountain peak detection
# ---------------------------
def _read_obj_peak_local(obj_path):
    """
    Returns (x,y,z) of the highest vertex (max z) in OBJ local coords.
    If OBJ missing or unreadable, returns None.
    """
    if not os.path.exists(obj_path):
        return None

    best = None  # (x,y,z)
    best_z = float("-inf")

    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        if z > best_z:
                            best_z = z
                            best = (x, y, z)
    except Exception:
        return None

    return best


def get_peak_world(shapes_dir, mountain_urdf, mountain_pos, mountain_orn):
    """
    Finds peak from the matching OBJ file and converts to world coordinates.
    If OBJ not found, returns a fallback peak (same as mountain_pos).
    """
    # assume OBJ has same base name as URDF
    base = os.path.splitext(mountain_urdf)[0]
    obj_name = base + ".obj"
    obj_path = os.path.join(shapes_dir, obj_name)

    peak_local = _read_obj_peak_local(obj_path)
    if peak_local is None:
        # fallback (still allows code to run)
        return (float(mountain_pos[0]), float(mountain_pos[1]), float(mountain_pos[2]))

    peak_world, _ = p.multiplyTransforms(
        mountain_pos, mountain_orn,
        peak_local, [0, 0, 0, 1]
    )
    return (float(peak_world[0]), float(peak_world[1]), float(peak_world[2]))


# ---------------------------
# Spawn helper
# ---------------------------
def spawn_point(radius=7.5, z=1.2):
    """
    Spawns around the mountain in a ring so the creature must climb inward/up.
    """
    a = random.uniform(0.0, 2.0 * math.pi)
    r = random.uniform(max(2.0, radius - 1.0), radius + 1.0)
    return (r * math.cos(a), r * math.sin(a), z)


# ---------------------------
# Main environment loader
# ---------------------------
def load_environment(
    arena_size=20,
    mountain_urdf="gaussian_pyramid.urdf",
    gravity=-10,
    mountain_position=(0, 0, -1),
    mountain_orientation_euler=(0, 0, 0),
):
    """
    Sets up the pybullet world: floor + arena walls + mountain.
    Returns:
        (floor_id, mountain_id, peak_world)
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, gravity)

    floor_id = make_arena(arena_size=arena_size)

    # Resolve shapes directory robustly (relative to this file)
    shapes_dir = os.path.join(os.path.dirname(__file__), "shapes")
    if not os.path.isdir(shapes_dir):
        # fallback if user runs from a different working directory
        shapes_dir = "shapes"
    p.setAdditionalSearchPath(shapes_dir)

    mountain_orn = p.getQuaternionFromEuler(mountain_orientation_euler)
    mountain_id = p.loadURDF(mountain_urdf, mountain_position, mountain_orn, useFixedBase=1)

    # Mountain friction helps climbing
    p.changeDynamics(mountain_id, -1, lateralFriction=1.2, rollingFriction=0.05, spinningFriction=0.05)

    peak_world = get_peak_world(
        shapes_dir=shapes_dir,
        mountain_urdf=mountain_urdf,
        mountain_pos=mountain_position,
        mountain_orn=mountain_orn,
    )

    return floor_id, mountain_id, peak_world
