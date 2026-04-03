"""
Real-Time Object Detection & Tracking
Uses YOLOv8 for detection + DeepSORT for multi-object tracking
"""

import cv2
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
CLASSES_TO_TRACK = None          # None = all COCO classes, or e.g. [0, 2, 5] (person, car, bus)
LOG_DIR = Path("logs")
COLORS = {}                      # per-class colors, generated lazily


def get_color(class_id: int) -> tuple:
    """Return a consistent BGR color for a given class ID."""
    if class_id not in COLORS:
        import random
        rng = random.Random(class_id * 7 + 13)
        COLORS[class_id] = (rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255))
    return COLORS[class_id]


def draw_box(frame, track_id: int, class_name: str, conf: float,
             x1: int, y1: int, x2: int, y2: int, color: tuple):
    """Draw bounding box + label on frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"#{track_id} {class_name} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, total_detections: int, class_counts: dict):
    """Draw FPS + detection summary HUD in top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 28 + 22 * (len(class_counts) + 1)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}   Objects: {total_detections}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 100), 1, cv2.LINE_AA)

    for i, (cls, count) in enumerate(class_counts.items()):
        cv2.putText(frame, f"  {cls}: {count}",
                    (8, 40 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)


def run(source, model_path: str = "yolov8n.pt", save_video: bool = False):
    """
    Main loop.

    Parameters
    ----------
    source      : int (webcam index) or str (video file path / RTSP URL)
    model_path  : YOLOv8 weights file
    save_video  : whether to write annotated output to disk
    """
    LOG_DIR.mkdir(exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"detections_{session_id}.csv"

    # ── Models ─────────────────────────────────────────────────────────────────
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

    # ── Video source ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_video:
        out_path = LOG_DIR / f"output_{session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (W, H))
        print(f"[+] Saving annotated video → {out_path}")

    # ── CSV logger ─────────────────────────────────────────────────────────────
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame", "track_id", "class", "confidence",
                         "x1", "y1", "x2", "y2"])
    print(f"[+] Logging detections → {log_path}")

    # ── Main loop ──────────────────────────────────────────────────────────────
    frame_idx = 0
    fps_timer = time.time()
    fps = 0.0

    print("\n[+] Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Stream ended.")
            break

        frame_idx += 1

        # ── Detect ─────────────────────────────────────────────────────────────
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                        classes=CLASSES_TO_TRACK)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # DeepSORT expects [left, top, width, height]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        # ── Track ──────────────────────────────────────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        class_counts = {}
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_id = track.det_class
            conf = track.det_conf or 0.0
            class_name = model.names.get(cls_id, str(cls_id)) if cls_id is not None else "?"
            l, t, r, b = map(int, track.to_ltrb())

            color = get_color(cls_id if cls_id is not None else 0)
            draw_box(frame, track_id, class_name, conf, l, t, r, b, color)

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            csv_writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                frame_idx, track_id, class_name, f"{conf:.3f}", l, t, r, b
            ])

        # ── HUD ────────────────────────────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps = frame_idx / (time.time() - fps_timer + 1e-9)
            fps_timer = time.time()
            frame_idx = 0

        draw_hud(frame, fps, sum(class_counts.values()), class_counts)

        if writer:
            writer.write(frame)

        cv2.imshow("Object Tracker  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    csv_file.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Session saved to {log_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Object Detection & Tracking")
    parser.add_argument("--source", default=0,
                        help="Webcam index (0) or path to video file / RTSP URL")
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                        help="YOLOv8 model size (n=fastest, m=most accurate)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output video to logs/")

    args = parser.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source
    run(src, model_path=args.model, save_video=args.save)