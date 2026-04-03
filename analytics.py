"""
analytics.py — Post-session analytics from logged CSV detections.

Usage:
    python analytics.py                        # auto-picks latest log
    python analytics.py logs/detections_X.csv  # specific file
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_log(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_rows(rows):
    class_counts = Counter()
    track_ids_per_class = defaultdict(set)
    conf_per_class = defaultdict(list)
    detections_over_time = []  # (frame, total_objects)
    frame_groups = defaultdict(list)

    for row in rows:
        cls = row["class"]
        tid = int(row["track_id"])
        conf = float(row["confidence"])
        frame = int(row["frame"])

        class_counts[cls] += 1
        track_ids_per_class[cls].add(tid)
        conf_per_class[cls].append(conf)
        frame_groups[frame].append(cls)

    # objects-per-frame timeline
    for frame, classes in sorted(frame_groups.items()):
        detections_over_time.append((frame, len(classes)))

    return class_counts, track_ids_per_class, conf_per_class, detections_over_time


def plot(path: Path):
    rows = load_log(path)
    if not rows:
        print("[!] Log file is empty.")
        return

    class_counts, track_ids_per_class, conf_per_class, timeline = parse_rows(rows)

    unique_counts = {cls: len(ids) for cls, ids in track_ids_per_class.items()}
    avg_conf = {cls: sum(v) / len(v) for cls, v in conf_per_class.items()}

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor="#0f0f12")
    fig.suptitle(f"Session Analytics  ·  {path.name}",
                 color="#e0e0e0", fontsize=13, y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.09)

    ax1 = fig.add_subplot(gs[0, 0])   # detections per class (bar)
    ax2 = fig.add_subplot(gs[0, 1])   # unique tracks per class (bar)
    ax3 = fig.add_subplot(gs[0, 2])   # avg confidence (horizontal bar)
    ax4 = fig.add_subplot(gs[1, :])   # timeline

    ACCENT = "#5dcaa5"
    BG = "#17171c"
    TEXT = "#c8c8d0"

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a35")

    classes = list(class_counts.keys())
    colors = plt.cm.Set2.colors

    # ── Bar: total detections ──────────────────────────────────────────────────
    bars = ax1.bar(classes, [class_counts[c] for c in classes],
                   color=colors[:len(classes)], edgecolor="none", width=0.6)
    ax1.set_title("Total detections", color=TEXT, fontsize=10)
    ax1.set_ylabel("count", color=TEXT, fontsize=8)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha="center", va="bottom",
                 color=TEXT, fontsize=8)

    # ── Bar: unique tracks ─────────────────────────────────────────────────────
    ax2.bar(classes, [unique_counts[c] for c in classes],
            color=colors[:len(classes)], edgecolor="none", width=0.6)
    ax2.set_title("Unique objects tracked", color=TEXT, fontsize=10)
    ax2.set_ylabel("unique IDs", color=TEXT, fontsize=8)

    # ── Horizontal bar: avg confidence ────────────────────────────────────────
    ax3.barh(classes, [avg_conf[c] * 100 for c in classes],
             color=ACCENT, edgecolor="none", height=0.5)
    ax3.set_xlim(0, 100)
    ax3.set_title("Avg confidence (%)", color=TEXT, fontsize=10)
    ax3.set_xlabel("%", color=TEXT, fontsize=8)
    for i, c in enumerate(classes):
        ax3.text(avg_conf[c] * 100 + 1, i, f"{avg_conf[c]:.0%}",
                 va="center", color=TEXT, fontsize=8)

    # ── Timeline ──────────────────────────────────────────────────────────────
    if timeline:
        frames, counts = zip(*timeline)
        ax4.fill_between(frames, counts, alpha=0.25, color=ACCENT)
        ax4.plot(frames, counts, color=ACCENT, linewidth=1.2)
        ax4.set_title("Objects in frame over time", color=TEXT, fontsize=10)
        ax4.set_xlabel("frame index", color=TEXT, fontsize=8)
        ax4.set_ylabel("object count", color=TEXT, fontsize=8)

    out = path.with_suffix(".png")
    plt.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"[+] Analytics chart saved → {out}")
    plt.show()


if __name__ == "__main__":
    log_dir = Path("logs")
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        logs = sorted(log_dir.glob("detections_*.csv"))
        if not logs:
            print("[!] No log files found in logs/. Run detector.py first.")
            sys.exit(1)
        target = logs[-1]
        print(f"[+] Using latest log: {target}")

    plot(target)