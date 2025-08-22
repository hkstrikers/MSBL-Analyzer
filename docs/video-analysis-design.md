# MSBL Video Analysis Library — Design Document

- Owner: hkstrikers / MSBL-Analyzer
- Last updated: 2025-08-21

## Summary

We will build a Python library that analyzes Mario Strikers Battle League (MSBL) gameplay videos to detect and track entities (characters, ball, items), identify actions (shots, passes, perfect pass/shot, turbo), and parse game state from the on-screen scoreboard. The default detector will be RF-DETR (open source), with YOLOv8 as a maintained alternative. The system outputs structured JSON/CSV timelines and an optional annotated video with overlays.

## Goals and Non‑Goals

Goals
- Ingest local video files (mp4/mkv/mov) and optionally streams.
- Detect key objects: characters, ball, items, UI widgets (scoreboard, timers, event banners).
- Track detected objects frame-to-frame and render overlays (boxes, IDs, trails).
- Infer actions/events: pass, shot, perfect pass, perfect shot, turbo; plus possession changes.
- Parse scoreboard to maintain game state: scores, clock, period, power-up slots.
- Attribute each character track to a team (home/away) using on-character markers and team colors.
- Export events.json, tracks.json, state_timeline.json, annotated.mp4.

Non‑Goals (initial)
- Network data integration, audio cues, player identity beyond character type.

## Contracts (Inputs/Outputs)

Inputs
- Video file: 30–60 FPS, 720p+ recommended.
- Optional config: model paths, thresholds, ROI templates.

Outputs
- events.json: list of detected events with timestamps and confidence.
- tracks.json: per-object trajectories and boxes.
- state_timeline.json: parsed scoreboard state over time.
- annotated.mp4: visualization with overlays.

Error modes
- Unsupported codec/resolution → explicit error.
- No scoreboard found → state_timeline entries omitted with reason.
- Low-confidence events → flagged and excluded from state updates unless threshold met.

## Architecture

Pipeline: Ingest → Object Detection → Multi-Object Tracking → Field/Possession → Event Detection → Scoreboard OCR → State Machine → Outputs/Render

### Ingestion
- OpenCV or PyAV for decoding and timestamps; configurable frame stride and batching.

### Object Detection
- RF-DETR (default), with YOLOv8/YOLOv9 as alternatives; classes: character, ball, item, scoreboard, event_banner, turbo_icon, team_marker.
- Transfer learning on custom MSBL dataset.
- Inference: per-class thresholds, FP16 on GPU when available.

### Tracking
- ByteTrack or OC-SORT (upgrade to StrongSORT if needed). ID smoothing; reinit on scene cuts.

### Field Coordinates & Possession
- Coarse homography via field lines or template; possession heuristic: nearest to ball with motion alignment and radius gating, plus last-possession tie-breaker.

### Team Attribution (Character → Team)
- Goal: For every character track, infer team: {home, away}.
- Signals:
  - On-character marker (glyph/shape) near the character’s head/feet: detect via `team_marker` class or local template match around the character box; compare shape to team markers near the team names on the scoreboard HUD.
  - Color match: extract dominant color of the on-character marker or kit accent; compare in HSV/LAB to color swatches next to team names in the scoreboard.
  - Spatial prior: teammates cluster during set plays; optional graph smoothing to enforce two clusters.
- Procedure (v1):
  1) Scoreboard parsing extracts two team reference templates: {shape_template, color_histogram} for left/right teams.
  2) For each character box, crop a small ROI at expected marker location (configurable offset relative to box) and compute:
     - Template similarity to each team’s shape_template (normalized cross-correlation / SSIM).
     - Color distance to each team’s color_histogram (Bhattacharyya / cosine) in HSV/LAB.
  3) Fuse scores with weights; assign team with max fused score if above threshold; else leave unknown.
  4) Temporal smoothing: majority vote over a window (e.g., 1–2 seconds) with confidence decay to prevent rapid flips.
- Edge cases: lighting shifts, overlapping characters, occluded markers → fall back to color-only or temporal prior; keep unknown until confident.

### Event Detection
- Hybrid rules + optional temporal model (SlowFast/TSM/X3D) on 1–2s clips.
- Heuristics v1:
  - pass: ball leaves A → travels toward B → B gains control within T2 and distance < D2.
  - shot: ball speed spike toward goal with goalie/goal interaction within T1.
  - perfect_*: event_banner "Perfect" near pass/shot start.
  - turbo: turbo_icon overlap or speed > V95 for N frames.

### Scoreboard Parsing (OCR)
- Detect scoreboard region → preprocess ROI → OCR via PaddleOCR/Tesseract with digit whitelist → debounce and fuse over time.
- Also extract team-side reference: capture marker shapes and color swatches adjacent to team names for team attribution.

### State Machine
- Maintains scores, period, clock, possession, power-ups; fuses events and OCR by confidence and recency.

### Outputs & Rendering
- JSON timelines and annotated video via OpenCV/FFmpeg; overlays include boxes, IDs, trails, and event markers.
- tracks.json to include `team` field when known: {"team":"home|away","confidence":float}; events may also include `teams` for involved actors.

## Data & Labeling
- Curate 1080p60 VODs; label character, ball, items, scoreboard, banners, turbo effects using CVAT/Label Studio; YOLO format; split by match.
- Add labels for `team_marker` near characters and capture scoreboard team markers/color patches to enable training and evaluation of team attribution.

## Models & Dependencies
- PyTorch + RF-DETR (default) or Ultralytics YOLO; ByteTrack/OC-SORT; PaddleOCR or Tesseract; OpenCV, PyAV; optional PyTorchVideo/MMAction2 for temporal.

### Detector choice and evaluation (Decision: RF-DETR default)

Recommendation (v1)
- Use RF-DETR as the default detector. It is open source, offers robust global reasoning that reduces duplicates in crowded scenes, and provides stable confidences that work well with multi-object trackers. Maintain YOLOv8 as an alternative for real-time scenarios.

Alternatives to consider later
- YOLOv8 (Ultralytics): faster and turnkey; strong small-object recall when tuned; great for near-real-time.
- YOLOv9/YOLOv10 (research/third-party repos): potential accuracy/speed gains; higher integration/maintenance cost.
- RT-DETR/RT-DETRv2: similar family; evaluate if implementations outperform RF-DETR for our data.
- YOLO-NAS / PP-YOLOE: also viable depending on hardware.

Key selection criteria for MSBL
- Small-object detection (the ball): AP_small, recall at low IoU; benefit from higher input size (e.g., 1280+).
- Latency at 1080p on GPU and CPU: frames/sec and per-frame ms including postprocess.
- Exportability and deployment: ONNX/TensorRT support, half precision, ease of use.
- Tracker synergy: stable class confidences and consistent box localization for ByteTrack/OC-SORT.

Training tips for the ball and HUD
- For RF-DETR: train with larger imgsz (e.g., 1280–1536), stronger small-object augmentation, and longer schedules.
- For YOLOv8: prefer P6 variants or larger imgsz; consider class-specific anchors.
- Use augmentations for motion blur, illumination changes; include negatives (no ball visible).
- If CPU-bound, consider tiling or dynamic high-res crops around play area.
- Class-specific thresholds: lower for ball (0.2–0.3) plus tracker filtering to reduce flicker.

Config hooks (sketch)
- detector:
  - name: rf_detr | yolo_v8
  - weights: path/to/best.pt
  - imgsz: 1280
  - conf: { ball: 0.25, character: 0.4, item: 0.35 }
  - iou: 0.6
  - device: auto|cpu|cuda:0
  - max_dets: 300

Evaluation plan
- Dataset: hold-out matches with diverse arenas; annotate ball/characters/items/HUD.
- Metrics: mAP@50 for classes with emphasis on ball AP and recall; FPS at 1080p; GPU and CPU configs.
- Tracking impact: Run ByteTrack using each detector’s outputs; report IDF1 and ID switches for ball and characters.
- Decision rule: Prefer RF-DETR if it meets ball recall targets with acceptable FPS for your workflow; otherwise switch to YOLOv8 for higher throughput.

License note
- RF-DETR is open source. Verify the repository’s LICENSE for compatibility with this project’s GPL-3.0 license before vendoring weights/code.

#### RF-DETR vs YOLOv8

Is RF-DETR a viable alternative? Yes, with trade-offs.

Pros (RF-DETR)
- Robust global reasoning from transformer architecture can reduce duplicate boxes and help in crowded/occluded scenes.
- Often stable confidence across frames, which can aid tracker association stability.
- Strong baseline accuracy without hand-tuned anchors.

Cons (RF-DETR)
- Latency is generally higher than YOLOv8 at comparable accuracy, especially on CPU; can be challenging to hit 1080p30 without a strong GPU.
- Small-object recall (the MSBL ball) can lag YOLO unless trained with higher resolution and targeted augmentation.
- Tooling/export less streamlined than Ultralytics (though ONNX export may be available, it’s less turnkey).

When to pick RF-DETR
- Your priority is precision in crowded scenes and you can afford lower FPS.
- YOLOv8 shows frequent ID switches or duplicate detections under heavy occlusion.

When to stick with YOLOv8
- You need real-time or near-real-time throughput.
- Ball recall and stability are paramount, and YOLOv8 already meets targets with P6/high imgsz.

Config hooks (RF-DETR sketch)
- detector:
  - name: rf_detr
  - weights: path/to/rf-detr/best.pt
  - imgsz: 1280
  - conf: { ball: 0.2, character: 0.4, item: 0.35 }
  - device: auto|cpu|cuda:0
  - max_dets: 300

Bake-off tips
- Train/evaluate both at imgsz 1280; for RF-DETR consider longer training and heavier augmentation for the ball.
- Profile end-to-end FPS including postprocessing and tracking.
- Compare per-frame ball recall and timing jitter; ensure trackers get stable inputs.

## Public API (Python)

- analyze_video(input_path: str, output_dir: str, config: Optional[Dict]) -> AnalysisResult
- load_models(config) -> Detector, Tracker, OCR
- draw_overlays(input_path, tracks, events, state_timeline, output_path)

Data types
- Box [x1,y1,x2,y2]; Track {track_id, class, boxes[(t, Box)], velocity};
- Event {type, t, actors[track_id], details, confidence}; StateSnapshot {t, scores, clock, period}.

## CLI (optional)

- msbl-analyze <video> --out out/ --annotated --config config.yaml

## Performance Targets

- 1080p30: real-time on mid-range GPU; 0.1–0.3x on CPU with YOLOv8n; memory < 4GB.

## Evaluation

- Detection mAP@50 (ball weighted higher); Tracking IDF1/MOTA; Events P/R/F1 with ±200ms tolerance; OCR accuracy for scores/clock.
- Team attribution: per-track accuracy and time-to-correct-label; percentage of frames with correct team; robustness under occlusion.

## Risks & Mitigations

- Small, fast ball: high-res crops, augment blur, class-specific anchors.
- HUD variability: train on variants; template matching fallback.
- Subtle perfect cues: banner detector + temporal model; allow manual correction later.
- Occlusion/crowding: StrongSORT + re-id; homography motion priors.

## Milestones

- M1: Detection + tracking + overlay (tracks.json, annotated.mp4).
- M2: Scoreboard OCR → state_timeline.json.
- M3: Rule-based pass/shot/turbo events.
- M4: Perfect pass/shot via banner detection.
- M5: Temporal model + homography improvements.
- M6: Packaging, CLI, docs, examples; release models.

## Interactive GUI Viewer

Purpose
- Allow users to load a video and interactively view analysis outputs (boxes, labels, trails, events, scoreboard state) with standard playback controls.

Key Features
- Open video file; optional auto-load of analysis outputs (tracks.json, events.json, state_timeline.json) from same folder.
- Play/Pause, seek slider, jump to event; adjustable playback rate; frame step.
- Overlays: toggle character/ball/items boxes and labels; show track IDs; event markers on timeline.
- Event panel listing detected actions; clicking seeks the video.
- Minimal latency rendering with OpenCV + PySide6 (Qt) on Windows; GPU not required.

Tech Choices
- Desktop: PySide6/Qt for native controls and performance; OpenCV for video decode; NumPy for frame manipulation.
- Optional Web: FastAPI backend serving frames + React front-end (future phase); not required for v1.

Data Contract
- The viewer consumes the same JSON outputs emitted by the analyzer:
  - tracks.json: [{track_id, class, boxes:[{t, b:[x1,y1,x2,y2]}], ...}]
  - events.json: [{type, t, actors, confidence, metadata}]
  - state_timeline.json: [{t, score, clock, period}]

Architecture
- QMainWindow with central video canvas (QLabel/QGraphicsView), docked EventList (QListView), and bottom transport bar (Play, Pause, Slider, Rate).
- Rendering loop via QTimer keyed to video timestamps (CAP_PROP_POS_MSEC).
- Overlay engine draws on OpenCV frames before converting to QImage; color-coded by class/team.
- Indexing: pre-index boxes and events by rounded timestamp bucket (e.g., 10–20 ms) for O(1) retrieval.

Performance Notes
- Avoid per-frame Python loops over all boxes by bucketing per time slice.
- Only render overlays visible in current frame; cache last rendered frame when slider idles.

Edge Cases
- Variable FPS/seek accuracy: use milliseconds positioning and tolerate ±1 frame mismatch.
- Missing JSON files: run in “video-only” mode with overlays disabled.
- Resolution mismatch: scale boxes if input was resized during analysis.

Deliverable (v1 minimal)
- `gui/app.py` with PySide6 viewer capable of opening a video and overlaying boxes/labels from JSON.
- `requirements.txt` listing PySide6, opencv-python, numpy.

## Proposed Repo Layout

- msbl_analyzer/
  - io/, models/, vision/, events/, configs/, cli.py
- data/, docs/, examples/, tests/

## JSON Schema Sketches

- events.json: [{"type":"pass","t":12.345,"actors":[21,7],"confidence":0.82}]
- tracks.json: [{"track_id":21,"class":"character","team":"home","team_conf":0.88,"boxes":[{"t":12.30,"b":[x1,y1,x2,y2]}]}]
- state_timeline.json: [{"t":0.0,"score":{"home":0,"away":0},"clock":"04:00"}]

## Next Steps

- Draft labeling guidelines and collect sample clips.
- Spin up RF-DETR + ByteTrack spike on a short video.
- Define config and dataclasses; create analyze_video stub with logging.
- Implement team attribution prototype: scoreboard team reference extraction, on-character marker ROI matching, color histogram comparison, and temporal smoothing.


# Resources
- [Roboflow training data](https://app.roboflow.com/strikers-iovql/msbl-analyzer-pdzo6/images/9C2WB8KL0dKvF6mUP9gF?jobStatus=assigned&annotationJob=CeEz2z3IxhR5clZhzaAa)