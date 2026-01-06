import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG (MATCHES YOUR FOLDERS)
# =========================

# Your experiments are stored in these folders (from your screenshot)
EXP3_ROOT = "results/Experiment3"
EXP4_ROOT = "results/Experiment4"
EXP5_ROOT = "results/Experiment5"

# --- Experiment 3: Population size ---
POP_GROUPS = {
    "Pop10": [os.path.join(EXP3_ROOT, "pop_size10", "gen_summary.csv")],
    "Pop20": [os.path.join(EXP3_ROOT, "pop_size20", "gen_summary.csv")],
    "Pop30": [os.path.join(EXP3_ROOT, "pop_size30", "gen_summary.csv")],
}
OUT_DIR_POP = os.path.join(EXP3_ROOT, "_compare_plots")

# --- Experiment 4: Mutation rate ---
MUT_GROUPS = {
    "Mut0.05": [os.path.join(EXP4_ROOT, "mutation_rate0.05", "gen_summary.csv")],
    "Mut0.18": [os.path.join(EXP4_ROOT, "mutation_rate0.18", "gen_summary.csv")],
    "Mut0.30": [os.path.join(EXP4_ROOT, "mutation_rate0.30", "gen_summary.csv")],
}
OUT_DIR_MUT = os.path.join(EXP4_ROOT, "_compare_plots")

# --- Experiment 5: Simulation steps ---
STEPS_GROUPS = {
    "Steps600":  [os.path.join(EXP5_ROOT, "sim_steps600", "gen_summary.csv")],
    "Steps1200": [os.path.join(EXP5_ROOT, "sim_steps1200", "gen_summary.csv")],
    "Steps2400": [os.path.join(EXP5_ROOT, "sim_steps2400", "gen_summary.csv")],
}
OUT_DIR_STEPS = os.path.join(EXP5_ROOT, "_compare_plots")

# Set True if you also want distance/progress/travel overlays
MAKE_EXTRA_METRIC_PLOTS = True


# =========================
# LOADING + AVERAGING
# =========================

def read_gen_summary(path):
    """Reads one gen_summary.csv into arrays."""
    gens = []
    best_fitness = []
    mean_fitness = []
    best_dist_xy = []
    best_progress_xy = []
    best_xy_travel = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row["generation"]))
            best_fitness.append(float(row["best_fitness"]))
            mean_fitness.append(float(row["mean_fitness"]))
            best_dist_xy.append(float(row["best_dist_xy"]))
            best_progress_xy.append(float(row["best_progress_xy"]))
            best_xy_travel.append(float(row["best_xy_travel"]))

    return {
        "gens": np.array(gens, dtype=int),
        "best_fitness": np.array(best_fitness, dtype=float),
        "mean_fitness": np.array(mean_fitness, dtype=float),
        "best_dist_xy": np.array(best_dist_xy, dtype=float),
        "best_progress_xy": np.array(best_progress_xy, dtype=float),
        "best_xy_travel": np.array(best_xy_travel, dtype=float),
    }


def average_curves(paths, key):
    """
    Average a curve across multiple gen_summary.csv files (seeds).
    Truncates to the shortest run length to align generations safely.
    Returns gens, mean_curve
    """
    curves = []
    gens_ref = None
    min_len = None

    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] Missing file: {p}")
            continue

        data = read_gen_summary(p)
        gens = data["gens"]
        curve = data[key]

        if gens_ref is None:
            gens_ref = gens
            min_len = len(gens)
        else:
            min_len = min(min_len, len(gens))

        curves.append(curve)

    if not curves:
        return None, None

    gens_ref = gens_ref[:min_len]
    curves = [c[:min_len] for c in curves]

    stacked = np.vstack(curves)
    avg = np.mean(stacked, axis=0)
    return gens_ref, avg


# =========================
# PLOTTING
# =========================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_fitness_two_panel(groups, out_dir, title):
    """
    Creates ONE figure like your screenshot:
    - top: best fitness vs generation (multiple lines)
    - bottom: mean fitness vs generation (multiple lines)
    """
    ensure_dir(out_dir)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # TOP: Best fitness
    ax = axes[0]
    ax.set_title("Average Best Fitness vs Generation", fontsize=14)
    for label, paths in groups.items():
        gens, avg_best = average_curves(paths, "best_fitness")
        if gens is None:
            continue
        ax.plot(gens, avg_best, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Best fitness")
    ax.legend()

    # BOTTOM: Mean fitness
    ax = axes[1]
    ax.set_title("Average Mean Fitness vs Generation", fontsize=14)
    for label, paths in groups.items():
        gens, avg_mean = average_curves(paths, "mean_fitness")
        if gens is None:
            continue
        ax.plot(gens, avg_mean, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Mean fitness")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, "compare_fitness.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] Saved: {out_path}")


def plot_overlay_metric(groups, out_dir, key, title, ylabel, filename):
    """Single-metric overlay (multi-line) plot."""
    ensure_dir(out_dir)

    plt.figure(figsize=(10, 6))
    plt.title(title)

    for label, paths in groups.items():
        gens, avg_curve = average_curves(paths, key)
        if gens is None:
            continue
        plt.plot(gens, avg_curve, label=label)

    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] Saved: {out_path}")


def make_all_plots(groups, out_dir, big_title):
    plot_fitness_two_panel(groups, out_dir, big_title)

    if MAKE_EXTRA_METRIC_PLOTS:
        plot_overlay_metric(
            groups, out_dir,
            key="best_dist_xy",
            title="Best Distance (XY) to Peak vs Generation (Comparison)",
            ylabel="Distance (m)",
            filename="compare_best_dist_xy.png"
        )
        plot_overlay_metric(
            groups, out_dir,
            key="best_progress_xy",
            title="Best Progress Toward Peak (XY) vs Generation (Comparison)",
            ylabel="Progress (m)",
            filename="compare_best_progress_xy.png"
        )
        plot_overlay_metric(
            groups, out_dir,
            key="best_xy_travel",
            title="Best Movement (XY travel) vs Generation (Comparison)",
            ylabel="XY travel (m)",
            filename="compare_best_xy_travel.png"
        )


def main():
    # Experiment 3: pop size
    make_all_plots(POP_GROUPS, OUT_DIR_POP, "Experiment 3 – Effect of Population Size")

    # Experiment 4: mutation rate
    make_all_plots(MUT_GROUPS, OUT_DIR_MUT, "Experiment 4 – Effect of Mutation Rate")

    # Experiment 5: sim steps
    make_all_plots(STEPS_GROUPS, OUT_DIR_STEPS, "Experiment 5 – Effect of Simulation Steps")


if __name__ == "__main__":
    main()
