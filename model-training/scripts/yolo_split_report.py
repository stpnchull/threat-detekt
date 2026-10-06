"""Report class balance across a YOLO train, val, and test split.

Point this script at a YOLO data yaml. It counts images and labels in
each split, then prints where the classes are thin and how many more
images would bring them closer together. It only reads the dataset.

    python yolo_split_report.py ./datasets/data.yaml

Requires the PyYAML package.

The yaml
train, val, and test may each be a folder, a .txt list of image paths,
a single image, or a list of those. A path: entry, when present, is the
dataset root. Relative split paths are resolved from that root.
Otherwise they are resolved from the folder that contains the yaml.
Absolute paths are used as written.

    path: ../datasets/threat
    train: images/train
    val: images/val
    test: images/test
    names:
      0: person
      1: handgun
      2: rifle
      3: shotgun
      4: knife

names may be a list or a mapping of class id to name. Ids that appear
in labels but not in names are shown as class_<id>. A missing test
entry is reported as an empty split. A split path that does not exist
is skipped, with a warning.

Images counted are .jpg, .jpeg, .png, .bmp, .tif, .tiff, and .webp.
Folders are scanned recursively. Pass --non-recursive to count only
the top level of each split folder.

Labels
For an image under an images folder, the label is the same path with
that folder renamed to labels and the extension changed to .txt:

    images/train/foo.jpg  ->  labels/train/foo.txt

When the path has no images folder, the label is the image with a
.txt extension in the same folder. A label line is a class id followed
by a box. Lines with fewer than five fields, or a class id that is not
a number, are skipped and a warning is printed.

How to read the report
Each split lists how many images were found, how many label files were
found, and how many labels are missing. Images with no label file stay
in the image total and add no objects.

The class table has two counts:

    Images    images that contain the class at least once
    Objects   boxes of that class

An image with three rifles counts as one rifle image and three rifle
objects.

Rebalance
The rebalance table combines every split and compares classes by image
count. RecommendedAdd is how many more images of that class would
reach the target. Collection priority lists those gaps, largest first.

    --target-strategy ratio    default. Target is --target-ratio of the
                               largest class, rounded up. The default
                               ratio is 0.7, so 70% of the largest class.
    --target-strategy max      Target is the largest class.
    --target-strategy fixed    Target is --target-count images per class.
                               The default count is 500.

Val and test
The last table suggests, per class, how many of that class's images
should sit in val and in test. The defaults are 15% val and 15% test,
with the rest in train. +Val and +Test are how many more images to
place in that split, from new data or by moving images out of train.
--min-val-per-class and --min-test-per-class raise those floors when
the class has enough images.

    --show-missing-labels    print each missing label path

Examples
Default report:

    python yolo_split_report.py ./datasets/data.yaml

List every image that has no label file:

    python yolo_split_report.py ./datasets/data.yaml --show-missing-labels

Match the largest class:

    python yolo_split_report.py ./datasets/data.yaml --target-strategy max

Aim for 500 images of each class:

    python yolo_split_report.py ./datasets/data.yaml \\
        --target-strategy fixed \\
        --target-count 500

Put about 20% of each class in val and 10% in test, with at least
20 images in each when the class is large enough:

    python yolo_split_report.py ./datasets/data.yaml \\
        --val-fraction 0.2 \\
        --test-fraction 0.1 \\
        --min-val-per-class 20 \\
        --min-test-per-class 20
"""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_entry(entry: Union[str, List[str], None]) -> List[str]:
    if entry is None:
        return []
    if isinstance(entry, list):
        return [str(x) for x in entry]
    return [str(entry)]


def resolve_path(base_path: Path, dataset_root: Optional[Path], entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        return entry_path

    if dataset_root is not None:
        return (dataset_root / entry).resolve()

    return (base_path / entry).resolve()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images_from_directory(directory: Path, recursive: bool = True) -> List[Path]:
    if not directory.exists():
        return []

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted([p for p in iterator if is_image_file(p)])


def collect_images_from_manifest_file(manifest_path: Path, base_dir: Path) -> List[Path]:
    if not manifest_path.exists():
        return []

    images = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            p = Path(line)
            if not p.is_absolute():
                p = (base_dir / p).resolve()

            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(p)

    return sorted(images)


def collect_split_images(
    entries: List[str],
    yaml_dir: Path,
    dataset_root: Optional[Path],
    recursive: bool = True,
) -> List[Path]:
    images: List[Path] = []

    for entry in entries:
        resolved = resolve_path(yaml_dir, dataset_root, entry)

        if resolved.is_dir():
            images.extend(collect_images_from_directory(resolved, recursive=recursive))
        elif resolved.is_file():
            if resolved.suffix.lower() == ".txt":
                images.extend(collect_images_from_manifest_file(resolved, resolved.parent))
            elif is_image_file(resolved):
                images.append(resolved)
        else:
            print(f"Warning: path does not exist and will be skipped: {resolved}")

    return sorted(set(images))


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def parse_label_file(label_path: Path) -> Tuple[Counter, List[int]]:
    object_counts = Counter()
    classes_in_image = set()

    if not label_path.exists():
        return object_counts, []

    with label_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"Warning: malformed label line in {label_path} at line {line_num}: {line}")
                continue

            try:
                class_id = int(float(parts[0]))
            except ValueError:
                print(f"Warning: invalid class id in {label_path} at line {line_num}: {parts[0]}")
                continue

            object_counts[class_id] += 1
            classes_in_image.add(class_id)

    return object_counts, sorted(classes_in_image)


def get_class_name_map(data: dict) -> Dict[int, str]:
    names = data.get("names", {})
    class_map = {}

    if isinstance(names, list):
        for idx, name in enumerate(names):
            class_map[idx] = str(name)
    elif isinstance(names, dict):
        for k, v in names.items():
            try:
                class_map[int(k)] = str(v)
            except ValueError:
                pass

    return class_map


def analyze_split(images: List[Path]) -> dict:
    object_counts = Counter()
    image_counts = Counter()
    missing_labels = []
    found_label_files = 0

    for image_path in images:
        label_path = infer_label_path(image_path)

        if label_path.exists():
            found_label_files += 1
        else:
            missing_labels.append(str(label_path))

        file_object_counts, classes_in_image = parse_label_file(label_path)

        object_counts.update(file_object_counts)
        for class_id in classes_in_image:
            image_counts[class_id] += 1

    return {
        "image_count": len(images),
        "label_count": found_label_files,
        "missing_label_count": len(missing_labels),
        "missing_labels": missing_labels,
        "object_counts": object_counts,
        "image_counts": image_counts,
    }


def merge_class_counts(*stats_dicts: dict) -> Tuple[Counter, Counter]:
    total_image_counts = Counter()
    total_object_counts = Counter()

    for stats in stats_dicts:
        total_image_counts.update(stats["image_counts"])
        total_object_counts.update(stats["object_counts"])

    return total_image_counts, total_object_counts


def get_all_known_class_ids(
    class_name_map: Dict[int, str],
    total_image_counts: Counter,
    total_object_counts: Counter,
) -> List[int]:
    all_ids = set(class_name_map.keys()) | set(total_image_counts.keys()) | set(total_object_counts.keys())
    return sorted(all_ids)


def compute_target_count(
    class_ids: List[int],
    image_counts: Counter,
    strategy: str,
    target_ratio: float,
    target_count: int,
) -> int:
    max_count = max((image_counts.get(cid, 0) for cid in class_ids), default=0)

    if strategy == "max":
        return max_count
    if strategy == "ratio":
        return math.ceil(max_count * target_ratio)
    if strategy == "fixed":
        return target_count

    raise ValueError(f"Unsupported strategy: {strategy}")


def print_split_report(split_name: str, stats: dict, class_name_map: Dict[int, str]) -> None:
    print(f"\n{split_name.upper()} SPLIT")
    print("-" * 70)
    print(f"Images found         : {stats['image_count']}")
    print(f"Label files found    : {stats['label_count']}")
    print(f"Missing label files  : {stats['missing_label_count']}")

    object_counts: Counter = stats["object_counts"]
    image_counts: Counter = stats["image_counts"]

    all_class_ids = sorted(set(class_name_map.keys()) | set(object_counts.keys()) | set(image_counts.keys()))
    if not all_class_ids:
        print("No class data found in this split.")
        return

    print("\nClass distribution:")
    print(f"{'Class ID':<10} {'Class Name':<25} {'Images':<10} {'Objects':<10}")
    print("-" * 70)

    for class_id in all_class_ids:
        class_name = class_name_map.get(class_id, f"class_{class_id}")
        print(
            f"{class_id:<10} "
            f"{class_name:<25} "
            f"{image_counts.get(class_id, 0):<10} "
            f"{object_counts.get(class_id, 0):<10}"
        )


def print_rebalance_summary(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    class_name_map: Dict[int, str],
    strategy: str,
    target_ratio: float,
    target_count: int,
) -> None:
    total_image_counts, total_object_counts = merge_class_counts(train_stats, val_stats, test_stats)
    all_class_ids = get_all_known_class_ids(class_name_map, total_image_counts, total_object_counts)

    print("\nREBALANCE SUMMARY (ALL SPLITS COMBINED)")
    print("-" * 110)

    if not all_class_ids:
        print("No class data found across any split.")
        return

    target = compute_target_count(all_class_ids, total_image_counts, strategy, target_ratio, target_count)

    if strategy == "max":
        strategy_desc = f"match largest class ({target})"
    elif strategy == "ratio":
        strategy_desc = f"reach {target_ratio:.0%} of largest class ({target})"
    else:
        strategy_desc = f"fixed minimum of {target}"

    print(f"Target strategy      : {strategy}")
    print(f"Target description   : {strategy_desc}")
    print()

    print(
        f"{'Class ID':<10} {'Class Name':<25} {'Images':<10} "
        f"{'Objects':<10} {'TargetImg':<10} {'RecommendedAdd':<15}"
    )
    print("-" * 110)

    for class_id in all_class_ids:
        class_name = class_name_map.get(class_id, f"class_{class_id}")
        img_count = total_image_counts.get(class_id, 0)
        obj_count = total_object_counts.get(class_id, 0)
        recommended_add = max(0, target - img_count)

        print(
            f"{class_id:<10} {class_name:<25} {img_count:<10} "
            f"{obj_count:<10} {target:<10} {recommended_add:<15}"
        )


def print_collection_priority(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    class_name_map: Dict[int, str],
    strategy: str,
    target_ratio: float,
    target_count: int,
) -> None:
    total_image_counts, total_object_counts = merge_class_counts(train_stats, val_stats, test_stats)
    all_class_ids = get_all_known_class_ids(class_name_map, total_image_counts, total_object_counts)
    target = compute_target_count(all_class_ids, total_image_counts, strategy, target_ratio, target_count)

    deficits = []
    for class_id in all_class_ids:
        current = total_image_counts.get(class_id, 0)
        deficit = max(0, target - current)
        deficits.append((deficit, class_id))

    deficits.sort(reverse=True)

    print("\nCOLLECTION PRIORITY")
    print("-" * 70)
    has_deficit = False
    for deficit, class_id in deficits:
        if deficit <= 0:
            continue
        has_deficit = True
        class_name = class_name_map.get(class_id, f"class_{class_id}")
        current = total_image_counts.get(class_id, 0)
        print(
            f"Class {class_id} ({class_name}): "
            f"current={current}, target={target}, gather≈{deficit} more images"
        )

    if not has_deficit:
        print("All classes already meet the selected rebalance target.")


def allocate_val_test_counts(
    total_images: int,
    val_fraction: float,
    test_fraction: float,
    min_val: int,
    min_test: int,
) -> Tuple[int, int]:
    """
    Return target (val_images, test_images) for a class with `total_images`
    containing that class, approximating val_fraction and test_fraction of total.

    Counts are capped so val + test <= total (remaining implied train).
    """
    if total_images <= 0:
        return 0, 0

    v = max(min_val, int(round(total_images * val_fraction)))
    t = max(min_test, int(round(total_images * test_fraction)))

    if v + t <= total_images:
        return v, t

    v = min(v, total_images)
    t = min(t, total_images)

    while v + t > total_images and v > min_val:
        v -= 1
    while v + t > total_images and t > min_test:
        t -= 1
    while v + t > total_images:
        if v >= t and v > 0:
            v -= 1
        elif t > 0:
            t -= 1
        else:
            denom = val_fraction + test_fraction
            if denom > 0:
                v = int(round(total_images * val_fraction / denom))
            else:
                v = total_images // 2
            v = max(0, min(total_images, v))
            t = total_images - v
            break

    return max(0, v), max(0, t)


def print_val_test_split_recommendations(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    class_name_map: Dict[int, str],
    val_fraction: float,
    test_fraction: float,
    min_val_per_class: int,
    min_test_per_class: int,
) -> None:
    """
    For each class, estimate how many *images containing that class* should sit in
    val vs test for a typical split, and how many more to add to each split given
    current counts.
    """
    train_ic: Counter = train_stats["image_counts"]
    val_ic: Counter = val_stats["image_counts"]
    test_ic: Counter = test_stats["image_counts"]

    total_image_counts, total_object_counts = merge_class_counts(
        train_stats, val_stats, test_stats
    )
    all_class_ids = get_all_known_class_ids(
        class_name_map, total_image_counts, total_object_counts
    )

    print("\nVAL / TEST SPLIT RECOMMENDATIONS (per class, image-level)")
    print("-" * 110)
    print(
        f"Assumes ~{val_fraction:.0%} of each class's images in val and "
        f"~{test_fraction:.0%} in test (rest train). "
        f"Min floors: val>={min_val_per_class}, test>={min_test_per_class} when possible."
    )
    print()

    header = (
        f"{'Class ID':<10} {'Class Name':<22} {'Total':<8} "
        f"{'ValNow':<8} {'ValTgt':<8} {'+Val':<8} "
        f"{'TestNow':<8} {'TestTgt':<8} {'+Test':<8}"
    )
    print(header)
    print("-" * 110)

    for class_id in all_class_ids:
        class_name = class_name_map.get(class_id, f"class_{class_id}")
        tr = train_ic.get(class_id, 0)
        va = val_ic.get(class_id, 0)
        te = test_ic.get(class_id, 0)
        total = tr + va + te

        val_tgt, test_tgt = allocate_val_test_counts(
            total,
            val_fraction,
            test_fraction,
            min_val_per_class,
            min_test_per_class,
        )

        add_val = max(0, val_tgt - va)
        add_test = max(0, test_tgt - te)

        print(
            f"{class_id:<10} {class_name:<22} {total:<8} "
            f"{va:<8} {val_tgt:<8} {add_val:<8} "
            f"{te:<8} {test_tgt:<8} {add_test:<8}"
        )

    print()
    print(
        "Notes: 'Total' counts images that contain the class at least once. "
        "'+Val' / '+Test' are additional images to place in that split (from new data "
        "or by moving from train) so the class is represented in val/test for stable metrics."
    )


def main() -> None:
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "yaml_path",
        type=Path,
        help="Path to the YOLO data yaml (data.yaml or config.yaml).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Count only the top level of each split directory.",
    )
    parser.add_argument(
        "--show-missing-labels",
        action="store_true",
        help="Print the path of each image that has no label file.",
    )
    parser.add_argument(
        "--target-strategy",
        choices=["max", "ratio", "fixed"],
        default="ratio",
        help=(
            "How the rebalance target is chosen: "
            "ratio (default, a fraction of the largest class), "
            "max (match the largest class), or "
            "fixed (a set image count per class)."
        ),
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.7,
        help=(
            "With --target-strategy ratio, the fraction of the largest "
            "class to aim for (default: 0.7). Must be > 0 and <= 1."
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=500,
        help=(
            "With --target-strategy fixed, the image count to aim for "
            "in each class (default: 500)."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help=(
            "Share of each class's images to place in val "
            "(default: 0.15). val-fraction + test-fraction must be < 1."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help=(
            "Share of each class's images to place in test "
            "(default: 0.15). val-fraction + test-fraction must be < 1."
        ),
    )
    parser.add_argument(
        "--min-val-per-class",
        type=int,
        default=0,
        help=(
            "Minimum val images to aim for per class when the class "
            "has enough images (default: 0)."
        ),
    )
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=0,
        help=(
            "Minimum test images to aim for per class when the class "
            "has enough images (default: 0)."
        ),
    )

    args = parser.parse_args()

    if args.target_strategy == "ratio" and not (0 < args.target_ratio <= 1.0):
        raise ValueError("--target-ratio must be > 0 and <= 1.0")

    if args.target_strategy == "fixed" and args.target_count < 0:
        raise ValueError("--target-count must be >= 0")

    if not (0 <= args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be >= 0 and < 1.0")
    if not (0 <= args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be >= 0 and < 1.0")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.0 (leave room for train)")
    if args.min_val_per_class < 0 or args.min_test_per_class < 0:
        raise ValueError("--min-val-per-class and --min-test-per-class must be >= 0")

    yaml_path = args.yaml_path.resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    data = load_yaml(yaml_path)
    yaml_dir = yaml_path.parent

    dataset_root = None
    if "path" in data and data["path"]:
        dataset_root = resolve_path(yaml_dir, None, str(data["path"]))

    class_name_map = get_class_name_map(data)

    train_entries = normalize_entry(data.get("train"))
    val_entries = normalize_entry(data.get("val"))
    test_entries = normalize_entry(data.get("test"))

    train_images = collect_split_images(
        train_entries,
        yaml_dir=yaml_dir,
        dataset_root=dataset_root,
        recursive=not args.non_recursive,
    )
    val_images = collect_split_images(
        val_entries,
        yaml_dir=yaml_dir,
        dataset_root=dataset_root,
        recursive=not args.non_recursive,
    )
    test_images = collect_split_images(
        test_entries,
        yaml_dir=yaml_dir,
        dataset_root=dataset_root,
        recursive=not args.non_recursive,
    )

    train_stats = analyze_split(train_images)
    val_stats = analyze_split(val_images)
    test_stats = analyze_split(test_images)

    print("\nYOLO DATASET ANALYSIS")
    print("=" * 70)
    print(f"YAML File            : {yaml_path}")
    print(f"Dataset Root         : {dataset_root if dataset_root else yaml_dir}")
    print(f"Train Images         : {train_stats['image_count']}")
    print(f"Val Images           : {val_stats['image_count']}")
    print(f"Test Images          : {test_stats['image_count']}")
    print("=" * 70)

    print_split_report("train", train_stats, class_name_map)
    print_split_report("val", val_stats, class_name_map)
    print_split_report("test", test_stats, class_name_map)

    print_rebalance_summary(
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        class_name_map=class_name_map,
        strategy=args.target_strategy,
        target_ratio=args.target_ratio,
        target_count=args.target_count,
    )

    print_collection_priority(
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        class_name_map=class_name_map,
        strategy=args.target_strategy,
        target_ratio=args.target_ratio,
        target_count=args.target_count,
    )

    print_val_test_split_recommendations(
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        class_name_map=class_name_map,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_val_per_class=args.min_val_per_class,
        min_test_per_class=args.min_test_per_class,
    )

    if args.show_missing_labels:
        for split_name, stats in [
            ("train", train_stats),
            ("val", val_stats),
            ("test", test_stats),
        ]:
            if stats["missing_labels"]:
                print(f"\nMissing labels in {split_name}:")
                for p in stats["missing_labels"]:
                    print(p)


if __name__ == "__main__":
    main()