"""Show a live camera view with person and weapon boxes that appear held.

Open a webcam, run the YOLO weights on its frames, and draw a person
and a weapon only when the two boxes look like the weapon is with that
person. Press q to close the window. Nothing is written to disk.

    python predict_video.py --model-path ./model/best-v1.pt

Requires the ultralytics and opencv-python packages.

What gets drawn
Class names from the model are matched without regard to case. The
person class is PERSON. The weapon classes are HANDGUN, RIFLE, SHOTGUN,
and KNIFE. Other classes are ignored.

A pair is treated as held when the boxes overlap, the weapon center
lies inside the person box, or the gap between the boxes is at most
--hold-distance-px (default 20 pixels). Both boxes in that pair are
drawn. A person with no weapon nearby is left off the frame, and so is
a weapon with no person nearby.

Person boxes are green. Weapon boxes are red. The label is the class
name and the confidence.

Confidence
YOLO first drops detections below --model-conf (default 0.25). The
person and weapon checks run after that. Both use --threshold (default
0.5) unless you set --person-threshold or --weapon-threshold. A
threshold below --model-conf cannot recover detections that YOLO
already dropped.

The camera
--camera-index chooses the camera (default 0). The window is named
Detections and defaults to 1280 by 720. That size is only for display.
Inference uses the camera frame at --imgsz (default 640).

Inference runs on a background thread. --process-every-n (default 1)
chooses how often a new frame is sent. Frames in between reuse the last
boxes. Only the newest waiting frame is kept. --show-performance-overlay
prints FPS, frame time, inference time, and how old those boxes are.

The default weights path is ./model/best-v1.pt in the current directory.

Examples
Default camera and weights:

    python predict_video.py --model-path ./model/best-v1.pt

Second camera, with the timing overlay:

    python predict_video.py \\
        --camera-index 1 \\
        --show-performance-overlay

Stricter weapons and a wider hold gap:

    python predict_video.py \\
        --person-threshold 0.4 \\
        --weapon-threshold 0.6 \\
        --hold-distance-px 30
"""

import argparse
import os
import threading
import time

import cv2
from ultralytics import YOLO

PERSON_CLASS = "PERSON"
WEAPON_CLASSES = {"HANDGUN", "RIFLE", "SHOTGUN", "KNIFE"}


def boxes_intersect(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_edge_distance(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return (dx**2 + dy**2) ** 0.5


def appears_held(person_box, weapon_box, max_distance_px):
    if boxes_intersect(person_box, weapon_box):
        return True
    if point_inside_box(box_center(weapon_box), person_box):
        return True
    return box_edge_distance(person_box, weapon_box) <= max_distance_px


def extract_target_detections(
    results,
    person_threshold,
    weapon_threshold,
    hold_distance_px,
):
    person_detections = []
    weapon_detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = results.names[int(class_id)].upper()
        if class_name != PERSON_CLASS and class_name not in WEAPON_CLASSES:
            continue

        if class_name == PERSON_CLASS:
            if score < person_threshold:
                continue
            person_detections.append((x1, y1, x2, y2, score, class_name))
        else:
            if score < weapon_threshold:
                continue
            weapon_detections.append((x1, y1, x2, y2, score, class_name))

    detections_to_draw = []
    for person in person_detections:
        person_box = person[:4]
        for weapon in weapon_detections:
            weapon_box = weapon[:4]
            if appears_held(person_box, weapon_box, hold_distance_px):
                detections_to_draw.append(person)
                detections_to_draw.append(weapon)

    return list({det for det in detections_to_draw})


def parse_args():
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model-path",
        default=os.path.join(".", "model", "best-v1.pt"),
        help="YOLO weights file (default: ./model/best-v1.pt).",
    )
    parser.add_argument(
        "-c",
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to open (default: 0).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Confidence for both person and weapon boxes "
            "when the specific thresholds are omitted (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=None,
        help="Confidence for PERSON boxes (default: --threshold).",
    )
    parser.add_argument(
        "--weapon-threshold",
        type=float,
        default=None,
        help="Confidence for weapon boxes (default: --threshold).",
    )
    parser.add_argument(
        "--model-conf",
        type=float,
        default=0.25,
        help=(
            "YOLO confidence cutoff applied before the person and weapon "
            "checks (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--model-iou",
        type=float,
        default=0.45,
        help="YOLO overlap cutoff for suppressing duplicate boxes (default: 0.45).",
    )
    parser.add_argument(
        "--hold-distance-px",
        type=float,
        default=20.0,
        help=(
            "Largest gap, in pixels, still treated as a held weapon "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=2,
        help="Box line thickness in pixels (default: 2).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Label text size (default: 0.6).",
    )
    parser.add_argument(
        "--text-thickness",
        type=int,
        default=1,
        help="Label text thickness in pixels (default: 1).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Smaller is faster (default: 640).",
    )
    parser.add_argument(
        "--process-every-n",
        type=int,
        default=1,
        help=(
            "Send a new frame for inference every N frames. "
            "Other frames reuse the last boxes (default: 1)."
        ),
    )
    parser.add_argument(
        "--show-performance-overlay",
        action="store_true",
        help="Draw FPS, frame time, inference time, and box age on the window.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Window width in pixels. Does not change inference (default: 1280).",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=720,
        help="Window height in pixels. Does not change inference (default: 720).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    person_threshold = args.person_threshold if args.person_threshold is not None else args.threshold
    weapon_threshold = args.weapon_threshold if args.weapon_threshold is not None else args.threshold

    model = YOLO(args.model_path)
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open webcam at index {args.camera_index}. "
            "Check your USB camera connection and index."
        )

    if args.display_width <= 0 or args.display_height <= 0:
        raise ValueError("--display-width and --display-height must be positive integers.")

    window_name = "Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.display_width, args.display_height)

    print("Press 'q' to quit.")
    frame_index = 0
    last_detections_to_draw = []
    last_inference_ms = 0.0
    last_inference_ts = 0.0
    smoothed_fps = 0.0
    fps_alpha = 0.15

    latest_inference_frame = None
    frame_lock = threading.Lock()
    result_lock = threading.Lock()
    stop_event = threading.Event()

    def inference_worker():
        nonlocal last_detections_to_draw, last_inference_ms, last_inference_ts, latest_inference_frame
        while not stop_event.is_set():
            frame_for_inference = None
            with frame_lock:
                if latest_inference_frame is not None:
                    frame_for_inference = latest_inference_frame
                    latest_inference_frame = None

            if frame_for_inference is None:
                time.sleep(0.001)
                continue

            inference_start = time.perf_counter()
            results = model(
                frame_for_inference,
                imgsz=args.imgsz,
                conf=args.model_conf,
                iou=args.model_iou,
                verbose=False,
            )[0]
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            detections = extract_target_detections(
                results=results,
                person_threshold=person_threshold,
                weapon_threshold=weapon_threshold,
                hold_distance_px=args.hold_distance_px,
            )
            with result_lock:
                last_detections_to_draw = detections
                last_inference_ms = inference_ms
                last_inference_ts = time.perf_counter()

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from webcam. Stopping.")
            break

        annotated_frame = frame.copy()

        run_inference = frame_index % max(1, args.process_every_n) == 0
        if run_inference:
            with frame_lock:
                latest_inference_frame = frame.copy()

        with result_lock:
            detections_snapshot = list(last_detections_to_draw)
            inference_latency_snapshot = last_inference_ms
            inference_age_ms = (
                (time.perf_counter() - last_inference_ts) * 1000.0 if last_inference_ts else 0.0
            )

        # Only draw overlapping person/weapon detections.
        for x1, y1, x2, y2, score, class_name in detections_snapshot:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{class_name} {score:.2f}"
            is_weapon = class_name in WEAPON_CLASSES
            color = (0, 0, 255) if is_weapon else (0, 255, 0)  # BGR: red for weapons, green for person
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                max(1, args.box_thickness),
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.1, args.font_scale),
                color,
                max(1, args.text_thickness),
                cv2.LINE_AA,
            )

        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        instantaneous_fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        if smoothed_fps == 0.0:
            smoothed_fps = instantaneous_fps
        else:
            smoothed_fps = (1 - fps_alpha) * smoothed_fps + fps_alpha * instantaneous_fps

        if args.show_performance_overlay:
            overlay_lines = [
                f"FPS: {smoothed_fps:.1f}",
                f"Frame Latency: {frame_ms:.1f} ms",
                f"Inference Latency: {inference_latency_snapshot:.1f} ms",
                f"Inference Age: {inference_age_ms:.1f} ms",
                f"Inference Every N Frames: {max(1, args.process_every_n)}",
            ]
            for idx, text in enumerate(overlay_lines):
                y = 25 + (idx * 24)
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        display_frame = cv2.resize(
            annotated_frame,
            (args.display_width, args.display_height),
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_index += 1

    stop_event.set()
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
