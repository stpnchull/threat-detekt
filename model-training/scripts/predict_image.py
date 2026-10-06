#!/usr/bin/env python3
"""Draw the best detection of each threat class on one image.

Point this script at an image and an output file. It runs the weights in
model/best-v10.pt, next to this script, and writes the image with one
box per class.

    python predict_image.py \\
        --image ./photos/scene.jpg \\
        --output ./predictions/scene.jpg

Requires the ultralytics and opencv-python packages.

Weights
The weights file is model/best-v10.pt beside this script. The drawn
classes are Person, Handgun, Rifle, Shotgun, and Knife. A detection is
kept when its class name matches one of those strings exactly.

Confidence
The script starts at a confidence of 0.5 and steps down through 0.4,
0.3, 0.2, 0.1, 0.05, and 0.01. It stops at the first level that returns
any detection from the model. Boxes below that chosen level are
dropped. When no level returns a detection, the script stops and writes
no image.

For each matching class, the highest-confidence box is drawn. The box
is green, with the class name and confidence on it. Other boxes of that
class are left off the image. When the model finds something else and
none of the named classes remain, the output image is still written,
with no boxes.

--output is the image file to write. Folders above that file are
created when they are missing.

Examples
Score one photo and write the marked copy:

    python predict_image.py \\
        --image ./photos/scene.jpg \\
        --output ./predictions/scene.jpg
"""

import argparse
import os

import cv2
from ultralytics import YOLO


def main() -> None:
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        required=True,
        dest="image_path",
        help="Image file to score.",
    )
    parser.add_argument(
        "--output",
        required=True,
        dest="output_path",
        help="Image file to write, with the boxes drawn on it.",
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    output_path = os.path.abspath(args.output_path)
    model_path = os.path.join(os.path.dirname(__file__), "model", "best-v10.pt")

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        raise SystemExit(1)

    if not os.path.isfile(image_path):
        print(f"Error: Image file not found: {image_path}")
        raise SystemExit(1)

    model = YOLO(model_path)

    # Drawn only when the model uses these exact class-name strings.
    target_classes = ["Person", "Handgun", "Rifle", "Shotgun", "Knife"]

    initial_threshold = 0.5
    threshold = initial_threshold
    confidence_levels = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

    results = None
    has_detections = False

    for conf_threshold in confidence_levels:
        print(f"Trying confidence threshold: {conf_threshold}")
        results = model.predict(
            source=image_path,
            imgsz=640,
            conf=conf_threshold,
            save=False,
            show=False,
        )

        has_detections = False
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                has_detections = True
                break

        if has_detections:
            threshold = conf_threshold
            print(f"Found detections with confidence threshold: {threshold}")
            break

        print(f"No detections found with confidence threshold: {conf_threshold}")

    if not has_detections:
        print(
            "No detections found even with minimum confidence threshold "
            f"({confidence_levels[-1]})"
        )
        print(
            "The image may not contain the target classes "
            "or the model needs more training"
        )
        raise SystemExit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        raise SystemExit(1)

    class_names = model.names

    # Keep the highest-confidence box for each target class.
    best_detections = {}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names[class_id]

            if class_name not in target_classes or confidence < threshold:
                continue

            if (
                class_name not in best_detections
                or confidence > best_detections[class_name]["confidence"]
            ):
                best_detections[class_name] = {
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": confidence,
                    "class_name": class_name,
                }

    for class_name, detection in best_detections.items():
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        cv2.rectangle(
            image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        print(f"Best detection for {class_name}: confidence {confidence:.2f}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, image)

    print(f"Prediction results saved to: {output_path}")
    print(f"Looking for {target_classes} with confidence >= {threshold}")
    print(
        f"Used confidence threshold: {threshold} "
        f"(reduced from initial {initial_threshold} if needed)"
    )
    print("Showing only the highest confidence detection for each class")
    print(f"Total classes detected: {len(best_detections)}")


if __name__ == "__main__":
    main()
