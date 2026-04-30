import re
import csv
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Picking-For-Robot")

DEBUG_DIR = PROJECT_ROOT / "debug_logs"

TENSORBOARD_DIRS = [
    PROJECT_ROOT / "logs_stage1",
    PROJECT_ROOT / "logs_stage2",
]

OUTPUT_DIR = PROJECT_ROOT / "training_report_output"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_MD = OUTPUT_DIR / "training_report_auto.md"

# Muốn chỉ vẽ Stage 2 thì để ["stage2"]
# Muốn vẽ tất cả thì để None
STAGES_TO_PROCESS = ["stage2"]
# STAGES_TO_PROCESS = None

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

def setup_report_style():
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",

        "lines.linewidth": 3.0,
    })


def save_clean(path: Path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


# ============================================================
# UTILS
# ============================================================

FLOAT_PATTERN = r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)"


def smooth(values, alpha=0.10):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return values

    for i in range(len(values)):
        if np.isnan(values[i]):
            values[i] = values[i - 1] if i > 0 else 0.0

    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]

    return smoothed


def safe_float(pattern, text, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def safe_int(pattern, text, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return int(m.group(1))
    except ValueError:
        return default


def safe_name(name: str):
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def rel_path(path: Path):
    return path.relative_to(OUTPUT_DIR).as_posix()


def should_process_stage(stage: str):
    if STAGES_TO_PROCESS is None:
        return True
    return stage in STAGES_TO_PROCESS


def detect_stage_substage_from_debug_name(path: Path):
    m = re.search(r"(stage\d+)_([0-9][A-Z])_debug\.log", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def detect_stage_substage_from_summary_name(path: Path):
    m = re.search(r"(stage\d+)_([0-9][A-Z])_summary\.log", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def detect_stage_substage_from_tb_path(path: Path):
    text = str(path.parent)

    m = re.search(r"(stage\d+).*?([0-9][A-Z])(?:_|\\|/|$)", text)
    if m:
        return m.group(1), m.group(2)

    m2 = re.search(r"([0-9][A-Z])", path.parent.name)
    if m2:
        sub = m2.group(1)
        return f"stage{sub[0]}", sub

    return "unknown_stage", "unknown_substage"


# ============================================================
# PARSE DEBUG LOG
# ============================================================

def parse_debug_log(debug_file: Path):
    """
    Parse Stage 2 debug format:
    [STEP] t=8 ep=1 step=1 sub=2A phase=descend hold=0 lift_hold=0
           r=-0.0798 dist=0.0571 xy=0.0027 z=0.0571 yaw=0.0007
           grip=0.0800 contact=(0,0) grasp=0 lift_dz=0.0000
           home_err=... success=0 truncated=0
    """

    episodes = {}

    with open(debug_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[STEP]" not in line:
                continue

            ep = safe_int(r"\bep=(\d+)", line)
            if ep is None:
                continue

            sub_match = re.search(r"\bsub=([0-9][A-Z])", line)
            substage = sub_match.group(1) if sub_match else "unknown"

            phase_match = re.search(r"\bphase=([A-Za-z0-9_]+)", line)
            phase = phase_match.group(1) if phase_match else "unknown"

            step = safe_int(r"\bstep=(\d+)", line, 0)
            reward = safe_float(r"\br=" + FLOAT_PATTERN, line, 0.0)

            dist = safe_float(r"\bdist=" + FLOAT_PATTERN, line)
            xy = safe_float(r"\bxy=" + FLOAT_PATTERN, line)
            z = safe_float(r"\bz=" + FLOAT_PATTERN, line)
            yaw = safe_float(r"\byaw=" + FLOAT_PATTERN, line)

            hold = safe_int(r"\bhold=(\d+)", line, 0)
            lift_hold = safe_int(r"\blift_hold=(\d+)", line, 0)
            home_hold = safe_int(r"\bhome_hold=(\d+)", line, 0)

            grip = safe_float(r"\bgrip=" + FLOAT_PATTERN, line)
            grasp = safe_int(r"\bgrasp=(\d+)", line, 0)
            lift_dz = safe_float(r"\blift_dz=" + FLOAT_PATTERN, line, 0.0)
            home_err = safe_float(r"\bhome_err=" + FLOAT_PATTERN, line, np.nan)

            contact_match = re.search(r"\bcontact=\((\d+),(\d+)\)", line)
            if contact_match:
                left_contact = int(contact_match.group(1))
                right_contact = int(contact_match.group(2))
            else:
                left_contact = 0
                right_contact = 0

            dual_contact = 1 if left_contact and right_contact else 0

            success = safe_int(r"\bsuccess=(\d+)", line, 0)
            truncated = safe_int(r"\btruncated=(\d+)", line, 0)

            if ep not in episodes:
                episodes[ep] = {
                    "episode": ep,
                    "substage": substage,

                    "total_reward": 0.0,
                    "success": 0,
                    "truncated": 0,

                    "final_dist": None,
                    "final_xy": None,
                    "final_z": None,
                    "final_yaw": None,
                    "final_grip": None,
                    "final_lift_dz": None,
                    "final_home_err": None,

                    "min_dist": None,
                    "min_xy": None,
                    "min_z": None,
                    "min_yaw": None,
                    "max_lift_dz": 0.0,
                    "min_home_err": None,

                    "max_hold": 0,
                    "max_lift_hold": 0,
                    "max_home_hold": 0,

                    "grasp_seen": 0,
                    "dual_contact_seen": 0,
                    "left_contact_seen": 0,
                    "right_contact_seen": 0,

                    "steps": 0,
                    "final_phase": phase,
                }

            row = episodes[ep]

            row["total_reward"] += reward
            row["success"] = max(row["success"], success)
            row["truncated"] = max(row["truncated"], truncated)

            row["steps"] = max(row["steps"], step)
            row["final_phase"] = phase

            row["max_hold"] = max(row["max_hold"], hold)
            row["max_lift_hold"] = max(row["max_lift_hold"], lift_hold)
            row["max_home_hold"] = max(row["max_home_hold"], home_hold)

            row["grasp_seen"] = max(row["grasp_seen"], grasp)
            row["dual_contact_seen"] = max(row["dual_contact_seen"], dual_contact)
            row["left_contact_seen"] = max(row["left_contact_seen"], left_contact)
            row["right_contact_seen"] = max(row["right_contact_seen"], right_contact)

            if dist is not None:
                row["final_dist"] = dist
                row["min_dist"] = dist if row["min_dist"] is None else min(row["min_dist"], dist)

            if xy is not None:
                row["final_xy"] = xy
                row["min_xy"] = xy if row["min_xy"] is None else min(row["min_xy"], xy)

            if z is not None:
                row["final_z"] = z
                row["min_z"] = z if row["min_z"] is None else min(row["min_z"], z)

            if yaw is not None:
                row["final_yaw"] = yaw
                row["min_yaw"] = yaw if row["min_yaw"] is None else min(row["min_yaw"], yaw)

            if grip is not None:
                row["final_grip"] = grip

            if lift_dz is not None:
                row["final_lift_dz"] = lift_dz
                row["max_lift_dz"] = max(row["max_lift_dz"], lift_dz)

            if home_err is not None and not np.isnan(home_err):
                row["final_home_err"] = home_err
                row["min_home_err"] = home_err if row["min_home_err"] is None else min(row["min_home_err"], home_err)

    return [episodes[ep] for ep in sorted(episodes.keys())]


# ============================================================
# PARSE SUMMARY LOG
# ============================================================

def parse_summary_log(summary_file: Path):
    """
    Parse Stage 2 summary format:
    [SUMMARY] t=152 sub=2A success_rate_100=1.000 reward_mean_100=10.642 ...
    """

    rows = []

    with open(summary_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[SUMMARY]" not in line:
                continue

            t = safe_int(r"\bt=(\d+)", line)
            sub_match = re.search(r"\bsub=([0-9][A-Z])", line)
            substage = sub_match.group(1) if sub_match else "unknown"

            row = {
                "t": t,
                "substage": substage,
                "success_rate_100": safe_float(r"\bsuccess_rate_100=" + FLOAT_PATTERN, line),
                "reward_mean_100": safe_float(r"\breward_mean_100=" + FLOAT_PATTERN, line),
                "ep_len_mean_100": safe_float(r"\bep_len_mean_100=" + FLOAT_PATTERN, line),
                "dist_mean_100": safe_float(r"\bdist_mean_100=" + FLOAT_PATTERN, line),
                "xy_mean_100": safe_float(r"\bxy_mean_100=" + FLOAT_PATTERN, line),
                "z_mean_100": safe_float(r"\bz_mean_100=" + FLOAT_PATTERN, line),
                "yaw_mean_100": safe_float(r"\byaw_mean_100=" + FLOAT_PATTERN, line),
                "grip_mean_100": safe_float(r"\bgrip_mean_100=" + FLOAT_PATTERN, line),
                "grasp_rate_100": safe_float(r"\bgrasp_rate_100=" + FLOAT_PATTERN, line),
                "dual_contact_rate_100": safe_float(r"\bdual_contact_rate_100=" + FLOAT_PATTERN, line),
                "lift_dz_mean_100": safe_float(r"\blift_dz_mean_100=" + FLOAT_PATTERN, line),
                "hold_mean_100": safe_float(r"\bhold_mean_100=" + FLOAT_PATTERN, line),
                "lift_hold_mean_100": safe_float(r"\blift_hold_mean_100=" + FLOAT_PATTERN, line),
                "home_hold_mean_100": safe_float(r"\bhome_hold_mean_100=" + FLOAT_PATTERN, line),
                "home_err_mean_100": safe_float(r"\bhome_err_mean_100=" + FLOAT_PATTERN, line),
            }

            rows.append(row)

    return rows


# ============================================================
# SAVE CSV
# ============================================================

def save_dict_rows(rows, out_file: Path):
    if not rows:
        return

    keys = list(rows[0].keys())

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


# ============================================================
# PLOT HELPERS
# ============================================================

def rolling_success(successes, window=10):
    out = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        out.append(sum(successes[start:i + 1]) / (i - start + 1))
    return out


def plot_line(x, y, title, xlabel, ylabel, label, path, alpha=0.10):
    setup_report_style()

    plt.figure(figsize=(12, 6))
    plt.plot(x, smooth(y, alpha=alpha), linewidth=3.0, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)
    save_clean(path)


# ============================================================
# PLOT DEBUG RESULTS
# ============================================================

def plot_debug_results(rows, stage, substage):
    setup_report_style()

    if not rows:
        return []

    out_dir = FIGURE_DIR / stage / substage
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [r["episode"] for r in rows]

    rewards = [r["total_reward"] for r in rows]
    successes = [r["success"] for r in rows]
    success_rate = smooth(rolling_success(successes, window=10), alpha=0.18)

    final_dist = [np.nan if r["final_dist"] is None else r["final_dist"] for r in rows]
    final_xy = [np.nan if r["final_xy"] is None else r["final_xy"] for r in rows]
    final_z = [np.nan if r["final_z"] is None else r["final_z"] for r in rows]
    final_yaw = [np.nan if r["final_yaw"] is None else r["final_yaw"] for r in rows]

    steps = [r["steps"] for r in rows]
    grip = [np.nan if r["final_grip"] is None else r["final_grip"] for r in rows]
    lift_dz = [np.nan if r["final_lift_dz"] is None else r["final_lift_dz"] for r in rows]
    max_lift_dz = [r["max_lift_dz"] for r in rows]

    grasp_rate = smooth(rolling_success([r["grasp_seen"] for r in rows], window=10), alpha=0.18)
    dual_contact_rate = smooth(rolling_success([r["dual_contact_seen"] for r in rows], window=10), alpha=0.18)

    home_err = [np.nan if r["final_home_err"] is None else r["final_home_err"] for r in rows]

    created = []

    # 01 Learning curve
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(episodes, smooth(rewards, alpha=0.10), label="Reward", linewidth=3.0)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"Learning Curve - {stage.upper()} {substage}", pad=15)
    ax1.grid(axis="y", linestyle="-", alpha=0.15)

    ax2 = ax1.twinx()
    ax2.plot(episodes, success_rate, label="Success Rate", linestyle="--", linewidth=3.0)
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="best", frameon=True)

    path = out_dir / "01_learning_curve_report.png"
    save_clean(path)
    created.append(path)

    # 02 Error reduction
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, smooth(final_dist, alpha=0.10), label="Total Distance", linewidth=3.0)
    plt.plot(episodes, smooth(final_xy, alpha=0.10), label="XY Error", linewidth=3.0)
    plt.plot(episodes, smooth(final_z, alpha=0.10), label="Z Error", linewidth=3.0)
    plt.xlabel("Episode")
    plt.ylabel("Distance (m)")
    plt.title(f"Error Reduction - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "02_error_reduction_report.png"
    save_clean(path)
    created.append(path)

    # 03 Success rate
    path = out_dir / "03_success_rate_report.png"
    plot_line(
        episodes,
        success_rate,
        f"Success Rate - {stage.upper()} {substage}",
        "Episode",
        "Success Rate",
        "Success Rate",
        path,
        alpha=0.25,
    )
    created.append(path)

    # 04 Episode length
    path = out_dir / "04_episode_length_report.png"
    plot_line(
        episodes,
        steps,
        f"Episode Length - {stage.upper()} {substage}",
        "Episode",
        "Steps",
        "Episode Length",
        path,
        alpha=0.12,
    )
    created.append(path)

    # 05 Yaw error
    path = out_dir / "05_yaw_error_report.png"
    plot_line(
        episodes,
        final_yaw,
        f"Yaw Error - {stage.upper()} {substage}",
        "Episode",
        "Yaw Error",
        "Yaw Error",
        path,
        alpha=0.10,
    )
    created.append(path)

    # 06 Grasp + contact rate
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, grasp_rate, label="Grasp Established Rate", linewidth=3.0)
    plt.plot(episodes, dual_contact_rate, label="Dual Contact Rate", linewidth=3.0)
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Grasp / Contact Quality - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "06_grasp_contact_report.png"
    save_clean(path)
    created.append(path)

    # 07 Grip width
    path = out_dir / "07_grip_width_report.png"
    plot_line(
        episodes,
        grip,
        f"Final Grip Width - {stage.upper()} {substage}",
        "Episode",
        "Grip Width",
        "Grip Width",
        path,
        alpha=0.10,
    )
    created.append(path)

    # 08 Lift delta
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, smooth(lift_dz, alpha=0.10), label="Final Lift ΔZ", linewidth=3.0)
    plt.plot(episodes, smooth(max_lift_dz, alpha=0.10), label="Max Lift ΔZ", linewidth=3.0)
    plt.xlabel("Episode")
    plt.ylabel("Lift Delta Z (m)")
    plt.title(f"Object Lift Progress - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "08_lift_delta_report.png"
    save_clean(path)
    created.append(path)

    # 09 Home error - mainly useful for 2D
    if substage == "2D" or not np.all(np.isnan(np.array(home_err, dtype=float))):
        path = out_dir / "09_home_error_report.png"
        plot_line(
            episodes,
            home_err,
            f"Home Error - {stage.upper()} {substage}",
            "Episode",
            "Home Error",
            "Home Error",
            path,
            alpha=0.10,
        )
        created.append(path)

    save_dict_rows(rows, out_dir / "debug_episode_metrics.csv")

    return created


# ============================================================
# PLOT SUMMARY RESULTS
# ============================================================

def plot_summary_results(rows, stage, substage):
    setup_report_style()

    if not rows:
        return []

    out_dir = FIGURE_DIR / stage / substage
    out_dir.mkdir(parents=True, exist_ok=True)

    t = [r["t"] for r in rows]
    created = []

    def get(name):
        return [np.nan if r.get(name) is None else r.get(name) for r in rows]

    # S01 summary learning
    plt.figure(figsize=(12, 6))
    plt.plot(t, smooth(get("reward_mean_100"), alpha=0.10), label="Reward Mean 100", linewidth=3.0)
    plt.plot(t, smooth(get("success_rate_100"), alpha=0.18), label="Success Rate 100", linewidth=3.0)
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.title(f"Summary Learning - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "S01_summary_learning_report.png"
    save_clean(path)
    created.append(path)

    # S02 summary errors
    plt.figure(figsize=(12, 6))
    plt.plot(t, smooth(get("dist_mean_100"), alpha=0.10), label="Dist Mean 100", linewidth=3.0)
    plt.plot(t, smooth(get("xy_mean_100"), alpha=0.10), label="XY Mean 100", linewidth=3.0)
    plt.plot(t, smooth(get("z_mean_100"), alpha=0.10), label="Z Mean 100", linewidth=3.0)
    plt.xlabel("Training Step")
    plt.ylabel("Distance")
    plt.title(f"Summary Error Metrics - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "S02_summary_error_report.png"
    save_clean(path)
    created.append(path)

    # S03 grasp/contact
    plt.figure(figsize=(12, 6))
    plt.plot(t, smooth(get("grasp_rate_100"), alpha=0.18), label="Grasp Rate 100", linewidth=3.0)
    plt.plot(t, smooth(get("dual_contact_rate_100"), alpha=0.18), label="Dual Contact Rate 100", linewidth=3.0)
    plt.xlabel("Training Step")
    plt.ylabel("Rate")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Summary Grasp / Contact - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "S03_summary_grasp_contact_report.png"
    save_clean(path)
    created.append(path)

    # S04 lift/home
    plt.figure(figsize=(12, 6))
    plt.plot(t, smooth(get("lift_dz_mean_100"), alpha=0.10), label="Lift ΔZ Mean 100", linewidth=3.0)

    home_err = get("home_err_mean_100")
    if not np.all(np.isnan(np.array(home_err, dtype=float))):
        plt.plot(t, smooth(home_err, alpha=0.10), label="Home Error Mean 100", linewidth=3.0)

    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.title(f"Summary Lift / Home Metrics - {stage.upper()} {substage}", pad=15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="-", alpha=0.15)

    path = out_dir / "S04_summary_lift_home_report.png"
    save_clean(path)
    created.append(path)

    save_dict_rows(rows, out_dir / "summary_metrics.csv")

    return created


# ============================================================
# PLOT TENSORBOARD
# ============================================================

def plot_tensorboard_file(tb_file: Path, stage, substage):
    setup_report_style()

    if not HAS_TENSORBOARD:
        print("⚠ Chưa cài tensorboard. Chạy: pip install tensorboard")
        return []

    out_dir = FIGURE_DIR / stage / substage
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ea = EventAccumulator(str(tb_file))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
    except Exception as e:
        print(f"⚠ Không đọc được TensorBoard: {tb_file}")
        print(e)
        return []

    useful_tags = [
        # PPO
        "train/explained_variance",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/approx_kl",
        "train/entropy_loss",

        # Custom callback
        "episode/success_rate_100",
        "episode/reward_mean_100",
        "episode/ep_len_mean_100",

        "grasp/xy_dist_mean_100",
        "grasp/z_dist_mean_100",
        "grasp/yaw_error_mean_100",
        "grasp/grip_width_mean_100",
        "grasp/dual_contact_rate_100",
        "grasp/grasp_established_rate_100",
        "grasp/lift_delta_mean_100",
        "grasp/hold_steps_mean_100",
        "grasp/lift_hold_steps_mean_100",
        "grasp/home_hold_steps_mean_100",
        "grasp/home_error_mean_100",
    ]

    created = []

    for tag in tags:
        if tag not in useful_tags:
            continue

        try:
            data = ea.Scalars(tag)
            steps = [x.step for x in data]
            values = [x.value for x in data]
        except Exception:
            continue

        if len(values) == 0:
            continue

        clean_title = tag.replace("train/", "").replace("episode/", "").replace("grasp/", "")
        clean_title = clean_title.replace("_", " ").title()
        tag_name = safe_name(tag)

        plt.figure(figsize=(12, 6))
        plt.plot(steps, smooth(values, alpha=0.18), linewidth=3.0, label=clean_title)
        plt.xlabel("Training Step")
        plt.ylabel(clean_title)
        plt.title(f"{clean_title} - {stage.upper()} {substage}", pad=15)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="-", alpha=0.15)

        path = out_dir / f"tb_{tag_name}_report.png"
        save_clean(path)
        created.append(path)

    return created


# ============================================================
# COPY SUMMARY
# ============================================================

def copy_summary_log(summary_file: Path, stage, substage):
    if not summary_file.exists():
        return None

    out_dir = FIGURE_DIR / stage / substage
    out_dir.mkdir(parents=True, exist_ok=True)

    target = out_dir / summary_file.name
    shutil.copy2(summary_file, target)
    return target


# ============================================================
# COMBINED PLOTS
# ============================================================

def plot_combined_stage(all_rows_by_stage_sub):
    setup_report_style()

    created = []
    grouped_by_stage = defaultdict(dict)

    for (stage, substage), rows in all_rows_by_stage_sub.items():
        grouped_by_stage[stage][substage] = rows

    for stage, sub_dict in grouped_by_stage.items():
        stage_dir = FIGURE_DIR / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        # reward comparison
        plt.figure(figsize=(12, 6))
        for substage, rows in sorted(sub_dict.items()):
            rewards = [r["total_reward"] for r in rows]
            x = list(range(1, len(rewards) + 1))
            plt.plot(x, smooth(rewards, alpha=0.10), label=substage, linewidth=3.0)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward Comparison - {stage.upper()}", pad=15)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="-", alpha=0.15)

        path = stage_dir / "all_substages_reward_comparison_report.png"
        save_clean(path)
        created.append(path)

        # success comparison
        plt.figure(figsize=(12, 6))
        for substage, rows in sorted(sub_dict.items()):
            successes = [r["success"] for r in rows]
            sr = rolling_success(successes, window=10)
            x = list(range(1, len(sr) + 1))
            plt.plot(x, smooth(sr, alpha=0.18), label=substage, linewidth=3.0)

        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Success Rate Comparison - {stage.upper()}", pad=15)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="-", alpha=0.15)

        path = stage_dir / "all_substages_success_comparison_report.png"
        save_clean(path)
        created.append(path)

        # grasp comparison
        plt.figure(figsize=(12, 6))
        for substage, rows in sorted(sub_dict.items()):
            gs = [r["grasp_seen"] for r in rows]
            gr = rolling_success(gs, window=10)
            x = list(range(1, len(gr) + 1))
            plt.plot(x, smooth(gr, alpha=0.18), label=substage, linewidth=3.0)

        plt.xlabel("Episode")
        plt.ylabel("Grasp Rate")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Grasp Rate Comparison - {stage.upper()}", pad=15)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="-", alpha=0.15)

        path = stage_dir / "all_substages_grasp_comparison_report.png"
        save_clean(path)
        created.append(path)

    return created


# ============================================================
# MARKDOWN REPORT
# ============================================================

def write_markdown_report(
    debug_plots,
    summary_plots,
    tb_plots,
    summary_logs,
    combined_plots,
    all_rows_by_stage_sub,
):
    lines = []

    lines.append("# Báo cáo biểu đồ huấn luyện Stage 2\n")
    lines.append("File này được tạo tự động từ `debug.log`, `summary.log` và TensorBoard logs.\n")

    if combined_plots:
        lines.append("\n---\n")
        lines.append("## So sánh tổng hợp giữa các substage\n")
        for path in combined_plots:
            lines.append(f"![{path.stem}]({rel_path(path)})\n")

    all_keys = sorted(set(
        list(debug_plots.keys())
        + list(summary_plots.keys())
        + list(tb_plots.keys())
    ))

    for stage, substage in all_keys:
        lines.append("\n---\n")
        lines.append(f"## {stage.upper()} - Substage {substage}\n")

        rows = all_rows_by_stage_sub.get((stage, substage), [])
        if rows:
            total_eps = len(rows)
            success_count = sum(r["success"] for r in rows)
            success_rate = success_count / total_eps if total_eps else 0.0
            mean_reward = sum(r["total_reward"] for r in rows) / total_eps if total_eps else 0.0
            grasp_count = sum(r["grasp_seen"] for r in rows)
            grasp_rate = grasp_count / total_eps if total_eps else 0.0

            lines.append("### Thống kê nhanh\n")
            lines.append(f"- Số episode: **{total_eps}**\n")
            lines.append(f"- Success rate: **{success_rate:.2%}**\n")
            lines.append(f"- Grasp rate: **{grasp_rate:.2%}**\n")
            lines.append(f"- Reward trung bình: **{mean_reward:.4f}**\n")
            lines.append(f"- Final distance cuối: **{rows[-1].get('final_dist')}**\n")
            lines.append(f"- Final grip cuối: **{rows[-1].get('final_grip')}**\n")
            lines.append(f"- Max lift dz cuối: **{rows[-1].get('max_lift_dz')}**\n")

        if (stage, substage) in summary_logs:
            lines.append("\n### Summary log\n")
            lines.append(f"- File summary đã copy: `{rel_path(summary_logs[(stage, substage)])}`\n")

        if (stage, substage) in debug_plots:
            lines.append("\n### Biểu đồ chính từ debug log\n")

            important_debug = [
                "01_learning_curve_report",
                "02_error_reduction_report",
                "03_success_rate_report",
                "04_episode_length_report",
                "06_grasp_contact_report",
                "07_grip_width_report",
                "08_lift_delta_report",
                "09_home_error_report",
            ]

            for name in important_debug:
                for path in debug_plots[(stage, substage)]:
                    if path.stem == name:
                        lines.append(f"![{path.stem}]({rel_path(path)})\n")

        if (stage, substage) in summary_plots:
            lines.append("\n### Biểu đồ từ summary log\n")

            important_summary = [
                "S01_summary_learning_report",
                "S02_summary_error_report",
                "S03_summary_grasp_contact_report",
                "S04_summary_lift_home_report",
            ]

            for name in important_summary:
                for path in summary_plots[(stage, substage)]:
                    if path.stem == name:
                        lines.append(f"![{path.stem}]({rel_path(path)})\n")

        if (stage, substage) in tb_plots:
            lines.append("\n### Biểu đồ TensorBoard\n")

            important_tb = [
                "tb_train_explained_variance_report",
                "tb_episode_success_rate_100_report",
                "tb_episode_reward_mean_100_report",
                "tb_grasp_dual_contact_rate_100_report",
                "tb_grasp_grasp_established_rate_100_report",
                "tb_grasp_lift_delta_mean_100_report",
                "tb_grasp_home_error_mean_100_report",
            ]

            for name in important_tb:
                for path in tb_plots[(stage, substage)]:
                    if path.stem == name:
                        lines.append(f"![{path.stem}]({rel_path(path)})\n")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🔍 Đang quét file Stage 2...")

    debug_files = sorted(DEBUG_DIR.glob("stage*_debug.log"))
    summary_files = sorted(DEBUG_DIR.glob("stage*_summary.log"))

    tb_files = []
    for tb_dir in TENSORBOARD_DIRS:
        if tb_dir.exists():
            tb_files.extend(sorted(tb_dir.rglob("events.out.tfevents.*")))

    print(f"📄 Debug logs tìm thấy: {len(debug_files)}")
    print(f"📄 Summary logs tìm thấy: {len(summary_files)}")
    print(f"📊 TensorBoard files tìm thấy: {len(tb_files)}")

    debug_plots = defaultdict(list)
    summary_plots = defaultdict(list)
    tb_plots = defaultdict(list)
    summary_logs = {}
    all_rows_by_stage_sub = {}

    # debug
    for debug_file in debug_files:
        stage, substage = detect_stage_substage_from_debug_name(debug_file)
        if stage is None or substage is None:
            continue
        if not should_process_stage(stage):
            continue

        print(f"➡ Parse debug: {stage} {substage} - {debug_file.name}")
        rows = parse_debug_log(debug_file)

        if not rows:
            print(f"⚠ Không parse được debug: {debug_file.name}")
            continue

        detected_sub = rows[0].get("substage")
        if detected_sub and detected_sub != "unknown":
            substage = detected_sub

        key = (stage, substage)
        all_rows_by_stage_sub[key] = rows

        created = plot_debug_results(rows, stage, substage)
        debug_plots[key].extend(created)

    # summary
    for summary_file in summary_files:
        stage, substage = detect_stage_substage_from_summary_name(summary_file)
        if stage is None or substage is None:
            continue
        if not should_process_stage(stage):
            continue

        print(f"➡ Parse summary: {stage} {substage} - {summary_file.name}")
        rows = parse_summary_log(summary_file)

        if not rows:
            print(f"⚠ Không parse được summary: {summary_file.name}")
            continue

        key = (stage, substage)
        copied = copy_summary_log(summary_file, stage, substage)
        if copied:
            summary_logs[key] = copied

        created = plot_summary_results(rows, stage, substage)
        summary_plots[key].extend(created)

    # tensorboard
    for tb_file in tb_files:
        stage, substage = detect_stage_substage_from_tb_path(tb_file)
        if not should_process_stage(stage):
            continue

        print(f"➡ Parse TensorBoard: {stage} {substage} - {tb_file.parent.name}")
        created = plot_tensorboard_file(tb_file, stage, substage)
        tb_plots[(stage, substage)].extend(created)

    combined_plots = plot_combined_stage(all_rows_by_stage_sub)

    write_markdown_report(
        debug_plots=debug_plots,
        summary_plots=summary_plots,
        tb_plots=tb_plots,
        summary_logs=summary_logs,
        combined_plots=combined_plots,
        all_rows_by_stage_sub=all_rows_by_stage_sub,
    )

    print("\n✅ DONE!")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📄 Markdown: {REPORT_MD}")


if __name__ == "__main__":
    main()