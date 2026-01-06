# === File: evolve.py ===
# Genetic algorithm to evolve creatures in the mountain environment
#
# Step 1: environment integration (cw_envt.load_environment + peak)
# Step 2: mountain fitness (height gain + XY closeness + XY progress + anti-flying)
# Step 3: deterministic spawn + per-generation summary CSV
# Step 4: auto-generate plots
# Step 5: replay best genome in GUI + capture a few screenshots
#
# IMPORTANT FIX:
#   If all fitness values are identical, your genome is not affecting motion.
#   This file now uses a genome-based JOINT CONTROLLER (sin waves),
#   instead of relying on creature.control_motors().

import os
import csv
import random
import time
import math

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

import creature
import genome
import cw_envt


# -------------------
# Config
# -------------------
GENE_COUNT = 4
POP_SIZE = 30
GENS = 30

SIM_HZ = 240
DT = 1.0 / SIM_HZ
SIM_STEPS = 1200            # longer gives time to show progress
MUTATION_RATE = 0.18
ELITE_KEEP = 2
IMMIGRANTS = 2
SEED = 42

# Mountain / spawn
ARENA_SIZE = 20
MOUNTAIN_URDF = "gaussian_pyramid.urdf"
SPAWN_RADIUS = 7.5

DETERMINISTIC_SPAWN = True
FIXED_SPAWN = (SPAWN_RADIUS, 0.0, 1.2)

# Fitness weights (tuned to show improvement)
W_HEIGHT = 0.25
W_CLOSENESS_XY = 1.0
W_PROGRESS_XY = 1.6
W_MOVE_XY = 0.25

AIRBORNE_PENALTY = 0.12
AIRBORNE_GRACE = 200

DEBUG_FITNESS = False

# Step 5: replay + screenshots
DO_REPLAY_AND_SCREENSHOTS = True
REPLAY_STEPS = 4001
SETTLE_BEFORE_SCREENSHOT = 60
CAPTURE_STEPS = [1000, 2000, 3000,4000]
SCREENSHOT_DIR = "screenshots"
REPLAY_REALTIME = False
REPLAY_SLEEP = 1 / 240.0

random.seed(SEED)
np.random.seed(SEED)


# -------------------
# Helpers
# -------------------
def _dist_xy(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def _apply_good_dynamics(robot_id):
    """Extra traction helps a LOT for early evolution."""
    num_joints = p.getNumJoints(robot_id)
    for link_idx in range(-1, num_joints):
        p.changeDynamics(
            robot_id,
            link_idx,
            lateralFriction=1.4,
            rollingFriction=0.03,
            spinningFriction=0.03,
            restitution=0.0,
        )


def _flatten_genome(g):
    # genome is list of numpy arrays; flatten to python floats
    flat = []
    for gene_arr in g:
        for v in gene_arr:
            flat.append(float(v))
    return flat


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def control_from_genome(robot_id, flat_params, step_i):
    """
    Turn the genome into sinusoidal joint targets:
      each joint uses 4 numbers: amplitude, phase, frequency, offset.
    This makes different genomes move differently -> fitness changes -> evolution improves.
    """
    num_joints = p.getNumJoints(robot_id)
    if num_joints <= 0:
        return

    t = step_i * DT
    two_pi = 2.0 * math.pi

    # Make sure we have params even if genome is short
    if len(flat_params) == 0:
        flat_params = [0.0]

    for j in range(num_joints):
        # take 4 params per joint (wrap around)
        base = (j * 4) % len(flat_params)
        a_raw = flat_params[base + 0 - (0 if base + 0 < len(flat_params) else len(flat_params))]
        p_raw = flat_params[base + 1 - (0 if base + 1 < len(flat_params) else len(flat_params))]
        f_raw = flat_params[base + 2 - (0 if base + 2 < len(flat_params) else len(flat_params))]
        o_raw = flat_params[base + 3 - (0 if base + 3 < len(flat_params) else len(flat_params))]

        # squash to stable ranges
        amp = 0.8 * math.tanh(a_raw)                     # [-0.8, 0.8]
        phase = two_pi * (abs(p_raw) % 1.0)              # [0, 2pi]
        freq = 0.5 + 2.0 * (abs(f_raw) % 1.0)            # [0.5, 2.5] Hz-ish
        offset = 0.3 * math.tanh(o_raw)                  # [-0.3, 0.3]

        target = offset + amp * math.sin(two_pi * freq * t + phase)

        info = p.getJointInfo(robot_id, j)
        j_lo = float(info[8])
        j_hi = float(info[9])

        # clamp if joint has valid limits
        if j_lo < j_hi and (j_lo > -1e9 and j_hi < 1e9):
            target = _clamp(target, j_lo, j_hi)

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=60,            # stronger so it can climb/turn
            positionGain=0.08,
            velocityGain=1.0,
        )


# -------------------
# Fitness
# -------------------
def evaluate_fitness(robot_id, cr, peak_world, gene, steps=SIM_STEPS):
    """
    We measure distance/progress in XY to the peak so it changes when the robot walks.
    """
    start_pos, _ = p.getBasePositionAndOrientation(robot_id)
    start_z = float(start_pos[2])

    peak_xy = (float(peak_world[0]), float(peak_world[1]))
    start_dist_xy = _dist_xy(start_pos, peak_xy)

    best_z = start_z
    best_dist_xy = start_dist_xy

    xy_travel = 0.0
    last_pos = start_pos
    final_pos = start_pos

    airborne_steps = 0

    flat = _flatten_genome(gene)

    for i in range(steps):
        control_from_genome(robot_id, flat, i)
        p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(robot_id)
        final_pos = pos
        cr.update_position(pos)

        # movement
        xy_travel += _dist_xy(pos, last_pos)
        last_pos = pos

        # height
        if pos[2] > best_z:
            best_z = float(pos[2])

        # closeness in XY
        dxy = _dist_xy(pos, peak_xy)
        if dxy < best_dist_xy:
            best_dist_xy = dxy

        # anti-flying
        if len(p.getContactPoints(bodyA=robot_id)) == 0:
            airborne_steps += 1

    end_dist_xy = _dist_xy(final_pos, peak_xy)
    progress_xy = start_dist_xy - end_dist_xy          # positive = moved toward peak

    height_gain = max(0.0, best_z - start_z)
    closeness_xy = 1.0 / (1.0 + best_dist_xy)

    airborne_excess = max(0, airborne_steps - AIRBORNE_GRACE)
    airborne_ratio = airborne_excess / float(steps) if steps > 0 else 0.0

    fitness = (
        W_HEIGHT * height_gain
        + W_CLOSENESS_XY * closeness_xy
        + W_PROGRESS_XY * progress_xy
        + W_MOVE_XY * xy_travel
        - AIRBORNE_PENALTY * airborne_ratio
    )

    if DEBUG_FITNESS:
        print(
            f"start_dxy={start_dist_xy:.3f} end_dxy={end_dist_xy:.3f} prog={progress_xy:.3f} "
            f"best_dxy={best_dist_xy:.3f} xy={xy_travel:.3f} hg={height_gain:.3f} "
            f"air={airborne_steps}/{steps} fit={fitness:.3f}"
        )

    return (
        float(fitness),
        float(height_gain),
        float(best_dist_xy),
        float(airborne_ratio),
        float(progress_xy),
        float(xy_travel),
    )


# -------------------
# Step 4: Plotting
# -------------------
def make_plots(summary_csv="gen_summary.csv", out_dir="plots"):
    if not os.path.exists(summary_csv):
        print(f"[plots] Missing {summary_csv}, skipping plots.")
        return
    os.makedirs(out_dir, exist_ok=True)

    gens, best_fit, mean_fit, best_dist, best_prog, best_xy = [], [], [], [], [], []

    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row["generation"]))
            best_fit.append(float(row["best_fitness"]))
            mean_fit.append(float(row["mean_fitness"]))
            best_dist.append(float(row["best_dist_xy"]))
            best_prog.append(float(row["best_progress_xy"]))
            best_xy.append(float(row["best_xy_travel"]))

    plt.figure()
    plt.plot(gens, best_fit, label="Best fitness")
    plt.plot(gens, mean_fit, label="Mean fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fitness_vs_generation.png"))
    plt.close()

    plt.figure()
    plt.plot(gens, best_dist, label="Best dist XY to peak")
    plt.xlabel("Generation")
    plt.ylabel("Distance (m)")
    plt.title("Best Distance (XY) to Peak vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distance_xy_vs_generation.png"))
    plt.close()

    plt.figure()
    plt.plot(gens, best_prog, label="Best progress XY")
    plt.xlabel("Generation")
    plt.ylabel("Progress (m)")
    plt.title("Progress Toward Peak (XY) vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "progress_xy_vs_generation.png"))
    plt.close()

    plt.figure()
    plt.plot(gens, best_xy, label="Best XY travel")
    plt.xlabel("Generation")
    plt.ylabel("XY travel (m)")
    plt.title("Movement (XY travel) vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "xy_travel_vs_generation.png"))
    plt.close()

    print(f"[plots] Saved plots to: {out_dir}/")


# -------------------
# Step 5: Screenshot capture
# -------------------
def _save_gui_screenshot(path, width=1280, height=720):
    cam = p.getDebugVisualizerCamera()
    view_mat = cam[2]
    proj_mat = cam[3]
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    img = np.reshape(rgba, (height, width, 4))[:, :, :3]
    plt.imsave(path, img)


def replay_and_capture(best_gene, steps=REPLAY_STEPS):
    if p.isConnected():
        p.disconnect()

    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.setTimeStep(DT)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    _, _, peak_world = cw_envt.load_environment(
        arena_size=ARENA_SIZE,
        mountain_urdf=MOUNTAIN_URDF
    )

    cr = creature.Creature(gene=best_gene)
    with open("test.udrf", "w") as f:
        f.write(cr.to_xml())

    spawn = FIXED_SPAWN if DETERMINISTIC_SPAWN else cw_envt.spawn_point(radius=SPAWN_RADIUS, z=1.2)
    robot_id = p.loadURDF("test.udrf", basePosition=spawn)
    _apply_good_dynamics(robot_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=10,
        cameraYaw=40,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0]
    )

    print(f"[replay] Capturing screenshots at steps: {CAPTURE_STEPS}")
    print(f"[replay] Saving screenshots to: {SCREENSHOT_DIR}/")

    flat = _flatten_genome(best_gene)

    for i in range(SETTLE_BEFORE_SCREENSHOT):
        control_from_genome(robot_id, flat, i)
        p.stepSimulation()
        if REPLAY_REALTIME:
            time.sleep(REPLAY_SLEEP)

    capture_set = set(CAPTURE_STEPS)

    for i in range(steps):
        control_from_genome(robot_id, flat, i)
        p.stepSimulation()

        if i in capture_set:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            path = os.path.join(SCREENSHOT_DIR, f"report_step_{i:04d}_z{pos[2]:.2f}.png")
            _save_gui_screenshot(path)
            print(f"[replay] Saved: {path}")

        if REPLAY_REALTIME:
            time.sleep(REPLAY_SLEEP)

    p.disconnect()


# -------------------
# Run one genome
# -------------------
def run_genome(g):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.setTimeStep(DT)

    _, _, peak_world = cw_envt.load_environment(
        arena_size=ARENA_SIZE,
        mountain_urdf=MOUNTAIN_URDF
    )

    cr = creature.Creature(gene=g)
    with open("test.udrf", "w") as f:
        f.write(cr.to_xml())

    spawn = FIXED_SPAWN if DETERMINISTIC_SPAWN else cw_envt.spawn_point(radius=SPAWN_RADIUS, z=1.2)
    robot_id = p.loadURDF("test.udrf", basePosition=spawn)
    _apply_good_dynamics(robot_id)

    if p.getNumJoints(robot_id) == 0:
        return 0.0, 0.0, 9999.0, 1.0, 0.0, 0.0

    return evaluate_fitness(robot_id, cr, peak_world, g, steps=SIM_STEPS)


# -------------------
# Evolution
# -------------------
def evolve(gui=False):
    if p.isConnected():
        p.disconnect()

    p.connect(p.GUI if gui else p.DIRECT)

    gene_spec = genome.Genome.get_gene_spec()
    gene_length = len(gene_spec)

    def new_random_genome():
        return genome.Genome.get_random_genome(gene_length, GENE_COUNT)

    population = [new_random_genome() for _ in range(POP_SIZE)]

    with open("fitness_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "generation", "creature",
            "fitness", "height_gain",
            "best_dist_xy", "airborne_ratio",
            "progress_xy", "xy_travel",
            "genome"
        ])

    with open("gen_summary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "generation",
            "best_fitness", "mean_fitness",
            "best_dist_xy", "best_progress_xy", "best_xy_travel",
            "best_airborne_ratio"
        ])

    best_overall_fit = -1e9
    best_overall_gene = None

    for gen in range(GENS):
        print(f"\n=== Generation {gen + 1} ===")

        results = [run_genome(g) for g in population]
        fitnesses = [r[0] for r in results]

        best_i = int(np.argmax(fitnesses))
        best_fit, best_hg, best_dxy, best_air, best_prog, best_xy = results[best_i]
        mean_fit = float(np.mean(fitnesses)) if fitnesses else 0.0

        print(
            f"BEST gen {gen+1}: fitness={best_fit:.3f} "
            f"dist_xy={best_dxy:.3f} progress_xy={best_prog:.3f} xy={best_xy:.3f} "
            f"air={best_air:.3f} | mean={mean_fit:.3f}"
        )

        if best_fit > best_overall_fit:
            best_overall_fit = best_fit
            best_overall_gene = population[best_i]

        with open("gen_summary.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                gen + 1,
                f"{best_fit:.6f}", f"{mean_fit:.6f}",
                f"{best_dxy:.6f}", f"{best_prog:.6f}", f"{best_xy:.6f}",
                f"{best_air:.6f}"
            ])

        with open("fitness_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            for i, (res, g) in enumerate(zip(results, population)):
                fit, hg, dxy, air, prog, xy = res
                genome_flat = [val for gene_arr in g for val in gene_arr]
                genome_str = "|".join(f"{float(v):.3f}" for v in genome_flat)

                print(f"Creature {i + 1} fitness: {fit:.3f}")
                writer.writerow([
                    gen + 1, i + 1,
                    f"{fit:.6f}", f"{hg:.6f}",
                    f"{dxy:.6f}", f"{air:.6f}",
                    f"{prog:.6f}", f"{xy:.6f}",
                    genome_str
                ])

        # --- selection + elitism
        sorted_idx = sorted(range(len(population)), key=lambda k: float(fitnesses[k]), reverse=True)
        sorted_genomes = [population[i] for i in sorted_idx]

        elites = sorted_genomes[:max(0, min(ELITE_KEEP, POP_SIZE))]
        parents = sorted_genomes[:max(2, POP_SIZE // 2)]

        # --- reproduction
        children = []
        children.extend(elites)

        while len(children) < POP_SIZE:
            if len(children) >= POP_SIZE - IMMIGRANTS:
                break

            p1, p2 = random.sample(parents, 2)
            child = []

            for g1, g2 in zip(p1, p2):
                new_gene = genome.Genome.crossover(g1, g2)

                if len(new_gene) < gene_length:
                    pad = genome.Genome.get_random_gene(gene_length - len(new_gene))
                    new_gene = np.concatenate([new_gene, pad])
                elif len(new_gene) > gene_length:
                    new_gene = new_gene[:gene_length]

                child.append(new_gene)

            child = genome.Genome.point_mutate(child, MUTATION_RATE)
            children.append(child)

        # random immigrants
        while len(children) < POP_SIZE:
            children.append(new_random_genome())

        population = children

    make_plots(summary_csv="gen_summary.csv", out_dir="plots")

    print("\n Evolution complete.")
    p.disconnect()

    if DO_REPLAY_AND_SCREENSHOTS and best_overall_gene is not None:
        replay_and_capture(best_overall_gene, steps=REPLAY_STEPS)

if __name__ == "__main__":
    evolve(gui=False)
