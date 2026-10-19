#!/usr/bin/env python3
"""Evaluate a trained YOLO model on a dataset split.

Prints overall and per-class precision, recall, mAP, fitness, and speed.
Use --split test for held-out testing conditions. Pass --split all to
evaluate train, val, and test and compare them side by side.

Examples
--------
Test-split evaluation (recommended for testing conditions)::

    python scripts/evaluate.py \\
        --weights runs/detect/train14/weights/best.pt \\
        --data config.yaml \\
        --name train14_test

All splits for a fuller picture of train / val / test behavior::

    python scripts/evaluate.py \\
        --weights runs/detect/train14/weights/best.pt \\
        --data config.yaml \\
        --split all \\
        --name train14_all_splits

Absolute paths also work::

    python scripts/evaluate.py \\
        --weights /path/to/best.pt \\
        --data /path/to/config.yaml \\
        --project runs/test \\
        --name train14_test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


SPLIT_CHOICES = ("train", "val", "test", "all")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_list(value: Any) -> list[Any]:
    """Convert list/tuple/ndarray-like values without boolean truth checks."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(value)


def collect_metrics(metrics) -> dict[str, Any]:
    """Extract overall and per-class detection metrics from a val result."""
    box = metrics.box
    names = metrics.names

    overall = {
        "precision": _as_float(box.mp),
        "recall": _as_float(box.mr),
        "mAP50": _as_float(box.map50),
        "mAP75": _as_float(box.map75),
        "mAP50-95": _as_float(box.map),
        "fitness": _as_float(box.fitness()),
    }

    per_class: list[dict[str, Any]] = []
    class_indices = _as_list(getattr(box, "ap_class_index", None))
    nt_per_class = getattr(box, "nt_per_class", None)
    nt_per_image = getattr(box, "nt_per_image", None)

    for i, class_id in enumerate(class_indices):
        precision, recall, ap50, ap = box.class_result(i)
        class_id = int(class_id)
        class_name = names[class_id] if isinstance(names, dict) else names[class_id]

        row: dict[str, Any] = {
            "class_id": class_id,
            "class": class_name,
            "precision": _as_float(precision),
            "recall": _as_float(recall),
            "mAP50": _as_float(ap50),
            "mAP50-95": _as_float(ap),
        }

        if nt_per_class is not None:
            row["instances"] = int(nt_per_class[class_id])
        if nt_per_image is not None:
            row["images"] = int(nt_per_image[class_id])

        per_class.append(row)

    speed = {
        key: _as_float(value)
        for key, value in _as_mapping(getattr(metrics, "speed", None)).items()
    }

    return {
        "overall": overall,
        "per_class": per_class,
        "speed_ms": speed,
        "results_dict": {
            key: _as_float(value)
            for key, value in _as_mapping(
                getattr(metrics, "results_dict", None)
            ).items()
        },
    }


def print_metrics_report(split: str, report: dict[str, Any]) -> None:
    """Print a human-readable metrics summary for one split."""
    overall = report["overall"]
    per_class = report["per_class"]
    speed = report["speed_ms"]

    print()
    print("=" * 72)
    print(f"EVALUATION SUMMARY  (split={split})")
    print("=" * 72)

    print()
    print("Overall:")
    print(f"  Precision: {_fmt(overall['precision'])}")
    print(f"  Recall:    {_fmt(overall['recall'])}")
    print(f"  mAP50:     {_fmt(overall['mAP50'])}")
    print(f"  mAP75:     {_fmt(overall['mAP75'])}")
    print(f"  mAP50-95:  {_fmt(overall['mAP50-95'])}")
    print(f"  Fitness:   {_fmt(overall['fitness'])}")

    print()
    print("Per-class:")
    header = (
        f"  {'Class':<12} {'Images':>7} {'Inst':>7} "
        f"{'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    if not per_class:
        print("  (no class metrics available)")
    else:
        for row in per_class:
            images = row.get("images", "-")
            instances = row.get("instances", "-")
            print(
                f"  {row['class']:<12} "
                f"{images:>7} "
                f"{instances:>7} "
                f"{_fmt(row['precision']):>7} "
                f"{_fmt(row['recall']):>7} "
                f"{_fmt(row['mAP50']):>7} "
                f"{_fmt(row['mAP50-95']):>9}"
            )

    if speed:
        print()
        print("Speed (ms / image):")
        for key in ("preprocess", "inference", "loss", "postprocess"):
            if key in speed:
                print(f"  {key:<12} {_fmt(speed[key])}")


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "   n/a"
    return f"{value:.4f}"


def save_reports(
    output_dir: Path,
    reports_by_split: dict[str, dict[str, Any]],
) -> None:
    """Write JSON and CSV summaries next to Ultralytics plot outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics_summary.json"
    json_path.write_text(
        json.dumps(reports_by_split, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "scope",
                "class_id",
                "class",
                "images",
                "instances",
                "precision",
                "recall",
                "mAP50",
                "mAP75",
                "mAP50-95",
                "fitness",
            ],
        )
        writer.writeheader()

        for split, report in reports_by_split.items():
            overall = report["overall"]
            writer.writerow(
                {
                    "split": split,
                    "scope": "overall",
                    "class_id": "",
                    "class": "all",
                    "images": "",
                    "instances": "",
                    "precision": overall["precision"],
                    "recall": overall["recall"],
                    "mAP50": overall["mAP50"],
                    "mAP75": overall["mAP75"],
                    "mAP50-95": overall["mAP50-95"],
                    "fitness": overall["fitness"],
                }
            )

            for row in report["per_class"]:
                writer.writerow(
                    {
                        "split": split,
                        "scope": "class",
                        "class_id": row["class_id"],
                        "class": row["class"],
                        "images": row.get("images", ""),
                        "instances": row.get("instances", ""),
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "mAP50": row["mAP50"],
                        "mAP75": "",
                        "mAP50-95": row["mAP50-95"],
                        "fitness": "",
                    }
                )

    print()
    print(f"Wrote metrics summary: {json_path}")
    print(f"Wrote metrics CSV:     {csv_path}")


def print_split_comparison(reports_by_split: dict[str, dict[str, Any]]) -> None:
    """Print a compact overall-metrics table across multiple splits."""
    if len(reports_by_split) < 2:
        return

    print()
    print("=" * 72)
    print("SPLIT COMPARISON (overall)")
    print("=" * 72)
    header = (
        f"  {'Split':<8} {'P':>7} {'R':>7} "
        f"{'mAP50':>7} {'mAP75':>7} {'mAP50-95':>9} {'Fit':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for split, report in reports_by_split.items():
        overall = report["overall"]
        print(
            f"  {split:<8} "
            f"{_fmt(overall['precision']):>7} "
            f"{_fmt(overall['recall']):>7} "
            f"{_fmt(overall['mAP50']):>7} "
            f"{_fmt(overall['mAP75']):>7} "
            f"{_fmt(overall['mAP50-95']):>9} "
            f"{_fmt(overall['fitness']):>7}"
        )


def evaluate_split(
    model: YOLO,
    data: Path,
    split: str,
    project: str,
    name: str,
    imgsz: int,
    batch: int,
) -> dict[str, Any]:
    """Run validation on one split and return a structured metrics report."""
    run_name = f"{name}_{split}"

    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        plots=True,
        verbose=True,
        project=project,
        name=run_name,
    )

    report = collect_metrics(metrics)
    print_metrics_report(split, report)
    return report


def evaluate(
    weights: Path,
    data: Path,
    project: str,
    name: str,
    imgsz: int,
    batch: int,
    split: str,
) -> None:
    """Run YOLO validation and print / save complete metrics."""
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if not data.exists():
        raise FileNotFoundError(f"Config not found: {data}")

    splits = ["train", "val", "test"] if split == "all" else [split]

    model = YOLO(str(weights))
    reports_by_split: dict[str, dict[str, Any]] = {}

    for current_split in splits:
        print()
        print(f"Evaluating split: {current_split}")
        reports_by_split[current_split] = evaluate_split(
            model=model,
            data=data,
            split=current_split,
            project=project,
            name=name,
            imgsz=imgsz,
            batch=batch,
        )

    print_split_comparison(reports_by_split)

    summary_dir = Path(project) / name
    save_reports(summary_dir, reports_by_split)

    print()
    print(f"Ultralytics plots/results written under: {summary_dir}")
    if split == "all":
        print(
            "Note: --split all also evaluates the train set. "
            "Treat train metrics as optimistic; use test for held-out performance."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained YOLO model with overall and per-class "
            "metrics on one split or all splits."
        )
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to trained YOLO .pt weights (e.g. runs/detect/train14/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to YOLO dataset config YAML (e.g. config.yaml)",
    )
    parser.add_argument(
        "--project",
        default="runs/test",
        help="Ultralytics project directory for evaluation outputs (default: runs/test)",
    )
    parser.add_argument(
        "--name",
        default="eval",
        help="Run name under --project (default: eval)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=SPLIT_CHOICES,
        help=(
            "Dataset split to evaluate: train, val, test, or all "
            "(default: test). Use test for held-out testing conditions."
        ),
    )

    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        data=args.data,
        project=args.project,
        name=args.name,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
    )


if __name__ == "__main__":
    main()
