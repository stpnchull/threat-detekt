#!/usr/bin/env python3
"""Remove empty YOLO labels and correct JPEG and PNG extensions.

Point this script at an image folder, a label folder, and a backup
folder. It copies both folders into the backup, deletes pairs whose
label file is empty, and renames images whose extension does not match
the file contents. It applies those changes without asking.

    python auto_correct_dataset.py \\
        --image-dir ./datasets/images \\
        --label-dir ./datasets/labels \\
        --backup-dir ./datasets/backup

Only files sitting directly in each folder are checked.

Backup
Before anything is changed, the image folder is copied to
<backup-dir>/new-images and the label folder to
<backup-dir>/new-labels. When the backup folder already exists, that
copy is skipped and the fixes still run. When the copy fails, the
script stops and leaves the dataset unchanged.

Empty labels
A .txt file with no text, or only whitespace, is empty. The script
looks for an image with the same name and a .jpg, .jpeg, .png, or
.bmp extension. When that image exists, both files are deleted. When
it does not, the empty label is left in place and a warning is printed.

Misnamed images
A file named .png whose contents are JPEG is renamed to .jpg. A file
named .jpg whose contents are PNG is renamed to .png. Other extensions
are left as they are.

Report
The script counts class ids on label lines that have five fields, then
prints that distribution before and after the fixes. Class ids are not
changed.

Examples
Correct one image folder and its labels:

    python auto_correct_dataset.py \\
        --image-dir ./datasets/images \\
        --label-dir ./datasets/labels \\
        --backup-dir ./datasets/backup
"""

import argparse
import glob
import os
import shutil
from collections import Counter


def find_empty_annotations(label_dir):
    """Find annotation files that are empty or contain only whitespace."""
    empty_files = []
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    for label_path in label_files:
        try:
            with open(label_path, "r") as f:
                content = f.read().strip()

            if not content:
                empty_files.append(label_path)
        except Exception as e:
            print(f"Error reading {label_path}: {e}")

    return empty_files


def find_misnamed_files(image_dir):
    """Find .png files that are JPEG data and .jpg files that are PNG data."""
    misnamed_files = []
    image_files = glob.glob(os.path.join(image_dir, "*"))

    for img_path in image_files:
        try:
            # Get actual file extension
            actual_ext = os.path.splitext(img_path)[1].lower()

            # Check if the file is actually a different format
            if actual_ext == ".png":
                # Check if it's actually JPEG
                with open(img_path, "rb") as f:
                    header = f.read(4)
                    if header.startswith(b"\xff\xd8\xff"):
                        misnamed_files.append((img_path, ".jpg"))
            elif actual_ext == ".jpg":
                # Check if it's actually PNG
                with open(img_path, "rb") as f:
                    header = f.read(8)
                    if header.startswith(b"\x89PNG\r\n\x1a\n"):
                        misnamed_files.append((img_path, ".png"))

        except Exception as e:
            print(f"Error checking {img_path}: {e}")

    return misnamed_files


def fix_empty_annotations(image_dir, label_dir, empty_files):
    """Delete each empty label and its matching image.

    The label is left in place when no matching image is found.
    """
    print(f"Fixing {len(empty_files)} empty annotation files...")

    removed_count = 0

    for label_path in empty_files:
        basename = os.path.splitext(os.path.basename(label_path))[0]

        # Find corresponding image file
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            potential_path = os.path.join(image_dir, f"{basename}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path:
            try:
                os.remove(image_path)
                os.remove(label_path)
                print(
                    f"Removed: {os.path.basename(image_path)} and {os.path.basename(label_path)}"
                )
                removed_count += 1
            except Exception as e:
                print(f"Error removing {basename}: {e}")
        else:
            print(
                f"Warning: No corresponding image found for {os.path.basename(label_path)}"
            )

    print(f"Successfully removed {removed_count} file pairs")
    return removed_count


def fix_misnamed_files(image_dir, misnamed_files):
    """Rename .png files that are JPEG data, and .jpg files that are PNG data."""
    print(f"Fixing {len(misnamed_files)} misnamed files...")

    fixed_count = 0

    for old_path, correct_ext in misnamed_files:
        new_path = os.path.splitext(old_path)[0] + correct_ext

        try:
            os.rename(old_path, new_path)
            print(
                f"Renamed: {os.path.basename(old_path)} -> {os.path.basename(new_path)}"
            )
            fixed_count += 1
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")

    print(f"Successfully fixed {fixed_count} file extensions")
    return fixed_count


def create_backup(image_dir, label_dir, backup_dir):
    """Copy image_dir and label_dir under backup_dir.

    When backup_dir already exists, skip the copy and report success.
    """
    print("Creating backup...")

    if os.path.exists(backup_dir):
        print(f"Backup directory already exists: {backup_dir}")
        return True

    try:
        shutil.copytree(image_dir, os.path.join(backup_dir, "new-images"))
        shutil.copytree(label_dir, os.path.join(backup_dir, "new-labels"))
        print(f"Backup created at: {backup_dir}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False


def analyze_class_distribution(label_dir):
    """Analyze the distribution of classes in annotations."""
    class_counts = Counter()
    valid_files = 0

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    for label_path in label_files:
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines or all(line.strip() == "" for line in lines):
                continue

            valid_files += 1
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                    except ValueError:
                        pass

        except Exception as e:
            print(f"Error reading {label_path}: {e}")

    return class_counts, valid_files


def main():
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Folder of images to check and correct.",
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        help=(
            "Folder of YOLO .txt labels that match those images by file name."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        required=True,
        help=(
            "Folder where copies are written before anything is changed. "
            "Images go in new-images/ and labels in new-labels/."
        ),
    )

    args = parser.parse_args()
    image_dir = os.path.abspath(args.image_dir)
    label_dir = os.path.abspath(args.label_dir)
    backup_dir = os.path.abspath(args.backup_dir)

    print("Automated Additional Training Data Fixer for YOLOv8")
    print("=" * 60)

    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return

    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found: {label_dir}")
        return

    # Create backup first
    if not create_backup(image_dir, label_dir, backup_dir):
        print("❌ Backup failed. Aborting fixes.")
        return

    # Find issues
    print("\nAnalyzing data...")
    empty_files = find_empty_annotations(label_dir)
    misnamed_files = find_misnamed_files(image_dir)
    class_dist, valid_files = analyze_class_distribution(label_dir)

    # Report initial state
    print("\nInitial state:")
    print(f"  Empty annotation files: {len(empty_files)}")
    print(f"  Misnamed files: {len(misnamed_files)}")
    print(f"  Valid annotation files: {valid_files}")
    print(f"  Class distribution: {dict(class_dist)}")

    # Apply fixes
    print("\nApplying fixes...")

    # Fix empty annotations
    if empty_files:
        removed_count = fix_empty_annotations(image_dir, label_dir, empty_files)
    else:
        print("No empty annotation files to fix.")
        removed_count = 0

    # Fix misnamed files
    if misnamed_files:
        fixed_count = fix_misnamed_files(image_dir, misnamed_files)
    else:
        print("No misnamed files to fix.")
        fixed_count = 0

    # Final analysis
    print("\nFinal analysis:")
    empty_files_after = find_empty_annotations(label_dir)
    misnamed_files_after = find_misnamed_files(image_dir)
    class_dist_after, valid_files_after = analyze_class_distribution(label_dir)

    print(
        f"  Empty annotation files: {len(empty_files_after)} (was {len(empty_files)})"
    )
    print(f"  Misnamed files: {len(misnamed_files_after)} (was {len(misnamed_files)})")
    print(f"  Valid annotation files: {valid_files_after} (was {valid_files})")
    print(f"  Class distribution: {dict(class_dist_after)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"✅ Removed {removed_count} empty annotation file pairs")
    print(f"✅ Fixed {fixed_count} misnamed files")
    print(f"✅ Backup created at: {backup_dir}")

    if not empty_files_after and not misnamed_files_after:
        print("\n🎉 All issues fixed! Data is ready for YOLOv8 training.")
    else:
        print("\n⚠️ Some issues remain:")
        if empty_files_after:
            print(f"  - {len(empty_files_after)} empty annotation files still exist")
        if misnamed_files_after:
            print(f"  - {len(misnamed_files_after)} misnamed files still exist")


if __name__ == "__main__":
    main()
