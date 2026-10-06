#!/usr/bin/env python3
"""Automatically label images and videos with a trained YOLO model.

Point this script at a directory of unlabeled images, videos, or both.
It runs the model, keeps detections that clear a per-class confidence
threshold, and writes a YOLO dataset you can use for further training.

    python auto_label.py \\
        --model ./weights/best.pt \\
        --source ./unlabeled \\
        --output ./datasets/auto_labeled

Requires the ultralytics and opencv-python packages.

Source files
The --source directory is scanned recursively. Images and videos in the
same tree are both labeled.

    Images   .jpg .jpeg .png .bmp .tif .tiff .webp
    Videos   .mp4 .avi .mov .mkv .m4v .webm .mpeg .mpg .wmv

What gets saved
An image or video frame is written only when it has at least one
accepted detection and no uncertain detections. Everything else is left
out of the dataset. Skipped media does not get an empty label file, so
it is not added as a negative example.

--output is created if needed. Two folders are written under it:

    images/    copied source images and extracted video frames
    labels/    one .txt file per saved image

Each label line is YOLO format. Box values are normalized from 0 to 1:

    class_id x_center y_center width height

Image files keep their original file name. Video frames are saved as
JPEG and named <video-name>_frame_<index>.jpg, where <index> is the
zero-based frame number in the source video. The output folders are
flat: the source subdirectory is not part of the file name, so two
source files that share a name replace each other.

Confidence bands
Each detection falls into one band. The cutoffs are CLASS_THRESHOLDS
in this file. On startup the script prints the bands for
every class in the model. Tune them against a human-labeled validation
set before trusting the labels.

    reject      below uncertain_min
                Dropped. A weak detection does not block the image.

    uncertain   from uncertain_min up to accept_min
                The whole image or frame is left out, so a partial
                label is never written.

    accept      at or above accept_min
                Written as a label line.

Classes omitted with --classes or --exclude-classes are ignored
completely. They produce no labels and do not put a frame in the
uncertain band.

While it runs you will see one line per image, and one line per kept or
uncertain video frame:

    [ACCEPT]     written to images/ and labels/
    [UNCERTAIN]  left out; a detection sat between the two thresholds
    [SKIP]       left out; nothing cleared the accept threshold
    [ERROR]      a video could not be opened, or a frame could not be saved

The closing summary counts accepted and skipped images. Skipped images
include both [UNCERTAIN] and [SKIP]. Video frames are counted separately
as accepted, uncertain, and skipped. Frames between stride steps are not
counted.

Frame sampling
--frame-stride N checks every Nth video frame. The default is 30, about
one frame per second for 30 FPS video. A smaller value keeps more frames
and takes longer. Image files are always checked.

Class filters
Pass --classes or --exclude-classes, not both. Names are matched without
regard to case and must be classes on the model.

    --classes handgun,rifle
        Label only those classes.

    --exclude-classes person,knife
        Label every other model class. At least one class must remain.

Examples
Images only:

    python auto_label.py \\
        --model ./weights/best.pt \\
        --source ./unlabeled/images \\
        --output ./datasets/auto_labeled

Videos, sampled at the default of every 30th frame:

    python auto_label.py \\
        --model ./weights/best.pt \\
        --source ./unlabeled/videos \\
        --output ./datasets/auto_labeled \\
        --frame-stride 30

Weapons only:

    python auto_label.py \\
        --model ./weights/best.pt \\
        --source ./unlabeled \\
        --output ./datasets/auto_labeled \\
        --classes handgun,rifle,shotgun,knife

Every class except people and knives:

    python auto_label.py \\
        --model ./weights/best.pt \\
        --source ./unlabeled \\
        --output ./datasets/auto_labeled \\
        --exclude-classes person,knife
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm",
    ".mpeg",
    ".mpg",
    ".wmv",
}


# -----------------------------------------------------------------------------
# Per-class confidence zones
#
# REJECT:
#     confidence < uncertain_min
#
# UNCERTAIN:
#     uncertain_min <= confidence < accept_min
#
# ACCEPT:
#     confidence >= accept_min
#
# IMPORTANT:
# These are example starting values only.
# Tune these against your human-labeled validation dataset.
# -----------------------------------------------------------------------------

# CLASS_THRESHOLDS = {
#     "person": {
#         "uncertain_min": 0.50,
#         "accept_min": 0.90,
#     },
#     "handgun": {
#         "uncertain_min": 0.50,
#         "accept_min": 0.95,
#     },
#     "rifle": {
#         "uncertain_min": 0.50,
#         "accept_min": 0.93,
#     },
#     "shotgun": {
#         "uncertain_min": 0.50,
#         "accept_min": 0.90,
#     },
#     "knife": {
#         "uncertain_min": 0.50,
#         "accept_min": 0.95,
#     },
# }

CLASS_THRESHOLDS = {
    "person": {
        "uncertain_min": 0.30,
        "accept_min": 0.75,
    },
    "handgun": {
        "uncertain_min": 0.30,
        "accept_min": 0.80,
    },
    "rifle": {
        "uncertain_min": 0.30,
        "accept_min": 0.80,
    },
    "shotgun": {
        "uncertain_min": 0.30,
        "accept_min": 0.80,
    },
    "knife": {
        "uncertain_min": 0.25,
        "accept_min": 0.75,
    },
}


# Used if the model contains a class that is not listed above.
DEFAULT_THRESHOLDS = {
    "uncertain_min": 0.50,
    "accept_min": 0.95,
}


# YOLO's own confidence filter.
#
# This MUST be lower than every uncertain_min value above.
#
# We intentionally allow low-confidence detections through YOLO so our
# per-class confidence logic can decide how to handle them.
INFERENCE_FLOOR = 0.05


# =============================================================================
# Utility Functions
# =============================================================================

def get_class_name(model: YOLO, class_id: int) -> str:
    """
    Return the class name associated with a YOLO class ID.
    """
    names = model.names

    if isinstance(names, dict):
        return names[class_id]

    return names[class_id]


def get_thresholds(class_name: str) -> dict[str, float]:
    """
    Return confidence thresholds for the supplied class.
    """
    return CLASS_THRESHOLDS.get(
        class_name,
        DEFAULT_THRESHOLDS,
    )


def model_class_names(model: YOLO) -> list[str]:
    """
    Return canonical class names from a loaded YOLO model.
    """
    names = model.names

    if isinstance(names, dict):
        return list(names.values())

    return list(names)


def build_class_name_lookup(
    model: YOLO,
) -> dict[str, str]:
    """
    Map lowercase class names to the model's canonical spelling.
    """
    lookup: dict[str, str] = {}

    for name in model_class_names(model):
        lookup[name.lower()] = name

    return lookup


def parse_class_list(raw: str) -> list[str]:
    """
    Parse a comma-separated list of class names from the CLI.
    """
    return [
        part.strip()
        for part in raw.split(",")
        if part.strip()
    ]


def resolve_allowed_classes(
    model: YOLO,
    include_classes: str | None,
    exclude_classes: str | None,
) -> frozenset[str] | None:
    """
    Determine which model classes participate in labeling.

    Returns None when every model class is active.

    Classes outside the allowed set are ignored completely: they do not
    produce labels and cannot trigger uncertain-frame rejection.
    """
    lookup = build_class_name_lookup(model)
    all_classes = frozenset(lookup.values())

    def resolve_one(name: str) -> str:
        key = name.lower()

        if key not in lookup:
            available = ", ".join(sorted(all_classes))
            raise ValueError(
                f"Unknown class '{name}'. "
                f"Model classes: {available}"
            )

        return lookup[key]

    if include_classes:
        requested = parse_class_list(include_classes)

        if not requested:
            raise ValueError(
                "--classes requires at least one class name."
            )

        return frozenset(
            resolve_one(name)
            for name in requested
        )

    if exclude_classes:
        requested = parse_class_list(exclude_classes)

        if not requested:
            raise ValueError(
                "--exclude-classes requires at least one class name."
            )

        excluded = frozenset(
            resolve_one(name)
            for name in requested
        )

        allowed = all_classes - excluded

        if not allowed:
            raise ValueError(
                "Cannot exclude every model class; "
                "at least one must remain active."
            )

        return allowed

    return None


def make_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """
    Create YOLO dataset output directories.

    output/
        images/
        labels/
    """
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return image_dir, label_dir


def validate_threshold_configuration(model: YOLO) -> None:
    """
    Validate threshold configuration before processing data.
    """
    print("\nModel classes and thresholds:")
    print("-" * 68)

    for class_id, class_name in model.names.items():
        thresholds = get_thresholds(class_name)

        uncertain_min = thresholds["uncertain_min"]
        accept_min = thresholds["accept_min"]

        if INFERENCE_FLOOR >= uncertain_min:
            raise ValueError(
                f"INFERENCE_FLOOR ({INFERENCE_FLOOR}) must be lower than "
                f"uncertain_min ({uncertain_min}) for class '{class_name}'."
            )

        if uncertain_min >= accept_min:
            raise ValueError(
                f"uncertain_min ({uncertain_min}) must be lower than "
                f"accept_min ({accept_min}) for class '{class_name}'."
            )

        print(
            f"{class_id:<3} "
            f"{class_name:<15} "
            f"reject < {uncertain_min:.2f} | "
            f"uncertain {uncertain_min:.2f}-{accept_min:.2f} | "
            f"accept >= {accept_min:.2f}"
        )

    print("-" * 68)
    print()


# =============================================================================
# Prediction Filtering
# =============================================================================

def filter_predictions(
    model: YOLO,
    result,
    allowed_classes: frozenset[str] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """
    Divide YOLO predictions into three confidence zones.

    Predictions whose class is not in allowed_classes are skipped entirely
    when allowed_classes is set.

    REJECT:
        confidence < uncertain_min

        The prediction is considered sufficiently weak that it is treated
        as noise and ignored.

    UNCERTAIN:
        uncertain_min <= confidence < accept_min

        The prediction could represent a legitimate object but is not
        reliable enough to become an automatic training annotation.

        If ANY uncertain prediction exists in an image/frame, the entire
        image/frame should be excluded from the pseudo-labeled dataset.

    ACCEPT:
        confidence >= accept_min

        The prediction is trusted and converted into a YOLO annotation.

    Returns:
        accepted_labels:
            YOLO formatted annotation strings.

        uncertain_predictions:
            Predictions falling inside the uncertain zone.

        rejected_predictions:
            Predictions falling below the uncertain threshold.
    """

    accepted_labels = []
    uncertain_predictions = []
    rejected_predictions = []

    if result.boxes is None or len(result.boxes) == 0:
        return (
            accepted_labels,
            uncertain_predictions,
            rejected_predictions,
        )

    class_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    # Normalized YOLO bounding boxes:
    #
    # x_center, y_center, width, height
    boxes = result.boxes.xywhn.cpu().tolist()

    for class_id_float, confidence, bbox in zip(
        class_ids,
        confidences,
        boxes,
    ):
        class_id = int(class_id_float)

        class_name = get_class_name(
            model,
            class_id,
        )

        if (
            allowed_classes is not None
            and class_name not in allowed_classes
        ):
            continue

        thresholds = get_thresholds(
            class_name,
        )

        uncertain_min = thresholds["uncertain_min"]
        accept_min = thresholds["accept_min"]

        prediction = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
        }

        # ---------------------------------------------------------------------
        # REJECT
        #
        # Prediction confidence is very low.
        #
        # Ignore it rather than allowing a weak model prediction to prevent
        # an otherwise high-quality image from entering the training dataset.
        # ---------------------------------------------------------------------

        if confidence < uncertain_min:
            rejected_predictions.append(
                prediction
            )

            continue

        # ---------------------------------------------------------------------
        # UNCERTAIN
        #
        # Prediction confidence is meaningful enough that it may represent
        # a real object, but isn't high enough to trust automatically.
        #
        # The caller should reject the ENTIRE image/frame.
        # ---------------------------------------------------------------------

        if confidence < accept_min:
            uncertain_predictions.append(
                prediction
            )

            continue

        # ---------------------------------------------------------------------
        # ACCEPT
        #
        # Convert the accepted detection to standard YOLO format:
        #
        # class_id x_center y_center width height
        # ---------------------------------------------------------------------

        x, y, width, height = bbox

        label = (
            f"{class_id} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        accepted_labels.append(label)

    return (
        accepted_labels,
        uncertain_predictions,
        rejected_predictions,
    )


# =============================================================================
# Logging Helpers
# =============================================================================

def record_accepted_class_counts(
    model: YOLO,
    accepted_labels: list[str],
    counts: Counter[str],
) -> None:
    """
    Increment per-class totals for accepted YOLO label lines.
    """
    for label in accepted_labels:
        class_id = int(label.split()[0])
        class_name = get_class_name(model, class_id)
        counts[class_name] += 1


def print_accepted_class_summary(
    counts: Counter[str],
) -> None:
    """
    Print totals of accepted annotations grouped by class name.
    """
    print()
    print("Accepted annotations by class:")

    if not counts:
        print("  (none)")
        return

    total = sum(counts.values())

    for class_name in sorted(counts.keys()):
        print(
            f"  {class_name:<12} {counts[class_name]}"
        )

    print(
        f"  {'Total':<12} {total}"
    )


def print_uncertain_predictions(
    source_name: str,
    uncertain_predictions: list[dict],
) -> None:
    """
    Print information about predictions that caused an image/frame to be
    rejected from the auto-labeled dataset.
    """
    print(
        f"[UNCERTAIN] {source_name} - "
        f"{len(uncertain_predictions)} uncertain detection(s)"
    )

    for prediction in uncertain_predictions:
        print(
            f"    "
            f"{prediction['class_name']:<10} "
            f"conf={prediction['confidence']:.3f}"
        )


# =============================================================================
# Image Processing
# =============================================================================

def process_image(
    model: YOLO,
    image_path: Path,
    image_output_dir: Path,
    label_output_dir: Path,
    accepted_class_counts: Counter[str],
    allowed_classes: frozenset[str] | None,
) -> bool:
    """
    Run automatic labeling on a single image.

    The image is accepted ONLY when:

        1. At least one detection is in the ACCEPT zone.

        2. No detection is in the UNCERTAIN zone.

    Low-confidence REJECT-zone detections are ignored.
    """

    results = model.predict(
        source=str(image_path),
        conf=INFERENCE_FLOOR,
        verbose=False,
    )

    result = results[0]

    (
        accepted,
        uncertain,
        rejected,
    ) = filter_predictions(
        model,
        result,
        allowed_classes,
    )

    # -------------------------------------------------------------------------
    # UNCERTAIN DETECTION
    #
    # Don't risk creating an incompletely annotated training image.
    # -------------------------------------------------------------------------

    if uncertain:
        print_uncertain_predictions(
            image_path.name,
            uncertain,
        )

        return False

    # -------------------------------------------------------------------------
    # NO ACCEPTED OBJECTS
    #
    # Do NOT automatically make this a negative training image.
    #
    # The absence of a high-confidence detection does not prove the image
    # truly contains none of the target classes.
    # -------------------------------------------------------------------------

    if not accepted:
        print(
            f"[SKIP] {image_path.name} - "
            f"no accepted detections"
        )

        return False

    # -------------------------------------------------------------------------
    # Write image and YOLO annotation
    # -------------------------------------------------------------------------

    output_image_path = (
        image_output_dir /
        image_path.name
    )

    output_label_path = (
        label_output_dir /
        f"{image_path.stem}.txt"
    )

    shutil.copy2(
        image_path,
        output_image_path,
    )

    output_label_path.write_text(
        "\n".join(accepted) + "\n",
        encoding="utf-8",
    )

    record_accepted_class_counts(
        model,
        accepted,
        accepted_class_counts,
    )

    print(
        f"[ACCEPT] {image_path.name} - "
        f"{len(accepted)} annotation(s)"
    )

    return True


# =============================================================================
# Video Processing
# =============================================================================

def process_video(
    model: YOLO,
    video_path: Path,
    image_output_dir: Path,
    label_output_dir: Path,
    frame_stride: int,
    accepted_class_counts: Counter[str],
    allowed_classes: frozenset[str] | None,
) -> tuple[int, int, int]:
    """
    Extract and auto-label frames from a video.

    frame_stride controls how often frames are evaluated.

    Example:

        30 FPS video
        frame_stride = 30

    evaluates approximately one frame per second.

    Returns:
        accepted_frames
        uncertain_frames
        skipped_frames
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        print(
            f"[ERROR] Could not open video: "
            f"{video_path}"
        )

        return 0, 0, 0

    accepted_frames = 0
    uncertain_frames = 0
    skipped_frames = 0

    frame_number = -1

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        # Skip frames according to requested sampling frequency.
        if frame_number % frame_stride != 0:
            continue

        results = model.predict(
            source=frame,
            conf=INFERENCE_FLOOR,
            verbose=False,
        )

        result = results[0]

        (
            accepted,
            uncertain,
            rejected,
        ) = filter_predictions(
            model,
            result,
            allowed_classes,
        )

        frame_name = (
            f"{video_path.stem}_"
            f"frame_{frame_number:08d}"
        )

        # ---------------------------------------------------------------------
        # UNCERTAIN
        #
        # Do not save the frame or any labels.
        # ---------------------------------------------------------------------

        if uncertain:
            uncertain_frames += 1

            print_uncertain_predictions(
                frame_name,
                uncertain,
            )

            continue

        # ---------------------------------------------------------------------
        # NO ACCEPTED OBJECTS
        #
        # Don't assume this is a valid negative training frame.
        # ---------------------------------------------------------------------

        if not accepted:
            skipped_frames += 1

            continue

        # ---------------------------------------------------------------------
        # ACCEPT
        # ---------------------------------------------------------------------

        image_path = (
            image_output_dir /
            f"{frame_name}.jpg"
        )

        label_path = (
            label_output_dir /
            f"{frame_name}.txt"
        )

        write_success = cv2.imwrite(
            str(image_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not write_success:
            print(
                f"[ERROR] Could not save frame: "
                f"{image_path}"
            )

            continue

        label_path.write_text(
            "\n".join(accepted) + "\n",
            encoding="utf-8",
        )

        record_accepted_class_counts(
            model,
            accepted,
            accepted_class_counts,
        )

        accepted_frames += 1

        print(
            f"[ACCEPT] {frame_name} - "
            f"{len(accepted)} annotation(s)"
        )

    capture.release()

    print()
    print(f"Video complete: {video_path.name}")
    print(f"  Accepted frames:  {accepted_frames}")
    print(f"  Uncertain frames: {uncertain_frames}")
    print(f"  Skipped frames:   {skipped_frames}")
    print()

    return (
        accepted_frames,
        uncertain_frames,
        skipped_frames,
    )


# =============================================================================
# Directory Processing
# =============================================================================

def process_directory(
    model_path: Path,
    source_dir: Path,
    output_dir: Path,
    frame_stride: int,
    include_classes: str | None = None,
    exclude_classes: str | None = None,
) -> None:
    """
    Process all supported images and videos within a directory recursively.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model does not exist: {model_path}"
        )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir}"
        )

    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"Source must be a directory: {source_dir}"
        )

    if frame_stride <= 0:
        raise ValueError(
            "--frame-stride must be greater than zero."
        )

    # -------------------------------------------------------------------------
    # Load YOLO model
    # -------------------------------------------------------------------------

    print(
        f"Loading YOLO model: {model_path}"
    )

    model = YOLO(
        str(model_path)
    )

    validate_threshold_configuration(
        model
    )

    allowed_classes = resolve_allowed_classes(
        model,
        include_classes,
        exclude_classes,
    )

    print("Active classes for labeling:")
    if allowed_classes is None:
        print(
            "  "
            + ", ".join(sorted(model_class_names(model)))
        )
    else:
        print(
            "  "
            + ", ".join(sorted(allowed_classes))
        )

    print()

    # -------------------------------------------------------------------------
    # Output structure
    # -------------------------------------------------------------------------

    (
        image_output_dir,
        label_output_dir,
    ) = make_output_dirs(
        output_dir
    )

    # -------------------------------------------------------------------------
    # Find source files recursively.
    # -------------------------------------------------------------------------

    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
    )

    image_files = [
        path
        for path in files
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    video_files = [
        path
        for path in files
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(
        f"Found {len(image_files)} image(s)"
    )

    print(
        f"Found {len(video_files)} video(s)"
    )

    print()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    accepted_images = 0
    rejected_images = 0

    accepted_video_frames = 0
    uncertain_video_frames = 0
    skipped_video_frames = 0

    accepted_class_counts: Counter[str] = Counter()

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------

    for image_path in image_files:

        success = process_image(
            model=model,
            image_path=image_path,
            image_output_dir=image_output_dir,
            label_output_dir=label_output_dir,
            accepted_class_counts=accepted_class_counts,
            allowed_classes=allowed_classes,
        )

        if success:
            accepted_images += 1
        else:
            rejected_images += 1

    # -------------------------------------------------------------------------
    # Videos
    # -------------------------------------------------------------------------

    for video_path in video_files:

        (
            video_accepted,
            video_uncertain,
            video_skipped,
        ) = process_video(
            model=model,
            video_path=video_path,
            image_output_dir=image_output_dir,
            label_output_dir=label_output_dir,
            frame_stride=frame_stride,
            accepted_class_counts=accepted_class_counts,
            allowed_classes=allowed_classes,
        )

        accepted_video_frames += video_accepted
        uncertain_video_frames += video_uncertain
        skipped_video_frames += video_skipped

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print("=" * 68)
    print("AUTO-LABELING COMPLETE")
    print("=" * 68)

    print()
    print("Images:")
    print(
        f"  Accepted: {accepted_images}"
    )
    print(
        f"  Skipped:  {rejected_images}"
    )

    print()
    print("Video frames:")
    print(
        f"  Accepted:  {accepted_video_frames}"
    )
    print(
        f"  Uncertain: {uncertain_video_frames}"
    )
    print(
        f"  Skipped:   {skipped_video_frames}"
    )

    print_accepted_class_summary(
        accepted_class_counts
    )

    print()
    print(
        f"Dataset written to: {output_dir}"
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to trained YOLO weights (.pt).",
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help=(
            "Directory of images, videos, or both. "
            "Subdirectories are included."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Dataset directory to create. "
            "Writes images/ and labels/ under it."
        ),
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=30,
        help=(
            "Check every Nth video frame "
            "(default: 30, about one frame per second at 30 FPS). "
            "Images are always checked."
        ),
    )

    class_group = parser.add_mutually_exclusive_group()

    class_group.add_argument(
        "--classes",
        metavar="NAMES",
        help=(
            "Comma-separated classes to label, for example "
            "handgun,rifle. Every other detection is ignored. "
            "Case-insensitive. Mutually exclusive with "
            "--exclude-classes."
        ),
    )

    class_group.add_argument(
        "--exclude-classes",
        metavar="NAMES",
        help=(
            "Comma-separated classes to ignore, for example "
            "person,knife. At least one model class must remain. "
            "Case-insensitive. Mutually exclusive with --classes."
        ),
    )

    args = parser.parse_args()

    process_directory(
        model_path=args.model,
        source_dir=args.source,
        output_dir=args.output,
        frame_stride=args.frame_stride,
        include_classes=args.classes,
        exclude_classes=args.exclude_classes,
    )


if __name__ == "__main__":
    main()