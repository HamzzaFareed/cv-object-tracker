# Real-Time Object Detection & Tracking

A production-style computer vision pipeline combining **YOLOv8** object detection with **DeepSORT** multi-object tracking, real-time analytics overlay, and automatic session logging.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Multi-object tracking** — persistent IDs across frames using DeepSORT (re-identifies objects after occlusion)
- **YOLOv8 detection** — state-of-the-art real-time detection on 80 COCO classes (person, car, bicycle, etc.)
- **Live HUD** — FPS counter + per-class object counts rendered directly on the video stream
- **Session logging** — every detection written to a timestamped CSV (frame, track ID, class, confidence, bounding box)
- **Analytics dashboard** — `analytics.py` generates charts: detection counts, unique tracks, confidence distribution, objects-over-time timeline
- **Flexible input** — works with webcam, local video files, and RTSP streams
- **Optional output recording** — save annotated video with `--save`

---

## Project Structure

```
cv_tracker/
├── detector.py       # Main detection + tracking loop
├── analytics.py      # Post-session analytics & charts
├── requirements.txt
├── logs/             # Auto-created; stores CSVs and output videos
└── README.md
```

---

## Setup

**1. Clone / download the project and create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

---

## Usage

### Run with webcam (default)
```bash
python detector.py
```

### Run on a video file
```bash
python detector.py --source path/to/video.mp4
```

### Use a larger, more accurate model
```bash
python detector.py --model yolov8s.pt
```

### Save annotated output video
```bash
python detector.py --source path/to/video.mp4 --save
```

### View analytics after a session
```bash
python analytics.py                         # auto-loads latest log
python analytics.py logs/detections_X.csv  # specific session
```

Press **Q** to quit the live window.

---

## How It Works

```
Frame from source
      │
      ▼
  YOLOv8 detection
  (bounding boxes + class + confidence)
      │
      ▼
  DeepSORT tracker
  (assigns persistent track IDs, handles occlusion via Kalman filter + Re-ID features)
      │
      ▼
  Annotated frame (drawn boxes, HUD)
      │
      ├─► Displayed in OpenCV window
      ├─► Optionally written to output video
      └─► Detection data appended to CSV log
```

---

## Model Options

| Model | Speed | mAP |
|-------|-------|-----|
| `yolov8n.pt` | Fastest (real-time on CPU) | 37.3 |
| `yolov8s.pt` | Fast | 44.9 |
| `yolov8m.pt` | Moderate | 50.2 |

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
