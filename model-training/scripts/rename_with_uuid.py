#!/usr/bin/env python3
"""Rename image and label pairs to hyphen-free UUID4 names.

Give this script a folder of training images that each have a matching
.txt label. It renames every pair in place so the image and its label
share a new 32-character id. File contents are left as they are.

    python rename_with_uuid.py ./datasets/auto_labeled

The script asks for confirmation before it changes any names. Answer
y or yes to continue. Any other answer leaves the files alone.

Layouts
Two layouts are recognized. The check is only one level deep.

A directory that contains both images/ and labels/ is treated as a
YOLO dataset. Images in images/ are paired with labels of the same
name in labels/:

    data/
      images/
        foo.jpg
      labels/
        foo.txt

Any other directory is treated as flat. Each image is paired with a
.txt file sitting next to it:

    data/
      foo.jpg
      foo.txt

Only files directly inside the image folder are considered. Files in
deeper subfolders are left unchanged.

What gets renamed
The image type is .jpg by default. Pass --type png to rename .png
files instead. The extension must be lowercase .jpg or .png. .jpeg
files are left unchanged.

A pair is renamed only when both files exist. The image and the label
receive the same new name:

    foo.jpg  ->  3f1c0a9e4b2d47c8a1e6d0b59f8a2c71.jpg
    foo.txt  ->  3f1c0a9e4b2d47c8a1e6d0b59f8a2c71.txt

An image with no matching .txt file is left unchanged, and a warning
is printed. A label with no matching image is left unchanged. Afterward
the script prints how many pairs were renamed.

If the generated name is already taken, another id is chosen.

Arguments
Pass the dataset directory as a positional argument or with
--directory. When both are given, the positional path is used. When
neither is given, the script uses the obj_train_data folder next to
this script.

    --type {jpg,png}    which image extension to rename (default: jpg)

Examples
Flat or YOLO dataset of JPEG images:

    python rename_with_uuid.py ./datasets/auto_labeled

The same folder, passed with the flag:

    python rename_with_uuid.py --directory ./datasets/auto_labeled

PNG images:

    python rename_with_uuid.py ./datasets/auto_labeled --type png

The default folder next to this script, using .jpg:

    python rename_with_uuid.py
"""

import argparse
import glob
import os
import uuid


def resolve_layout(data_dir):
    """
    Detect whether data_dir is a YOLO images/labels layout or a flat directory.

    Returns:
        tuple[str, str]: (image_dir, label_dir)
    """
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    if os.path.isdir(image_dir) and os.path.isdir(label_dir):
        return image_dir, label_dir

    return data_dir, data_dir


def rename_files_with_uuid(data_dir, image_ext):
    """
    Rename all image/.txt file pairs with UUID4 values (without hyphens).

    Args:
        data_dir (str): Path to a flat directory, or a parent containing
            images/ and labels/ subdirectories
        image_ext (str): Image extension without dot, "png" or "jpg"
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        return

    image_dir, label_dir = resolve_layout(data_dir)

    if image_dir == label_dir:
        print(f"Layout: flat directory ({image_dir})")
    else:
        print(f"Layout: separate trees")
        print(f"  Images: {image_dir}")
        print(f"  Labels: {label_dir}")

    pattern = os.path.join(image_dir, f"*.{image_ext}")
    image_files = glob.glob(pattern)

    renamed_count = 0
    total_files = len(image_files)

    print(f"Found {total_files} .{image_ext} files to rename...")

    rename_mapping = {}

    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        txt_file = os.path.join(label_dir, base_name + ".txt")

        if os.path.exists(txt_file):
            new_name = str(uuid.uuid4()).replace("-", "")

            while (
                os.path.exists(
                    os.path.join(image_dir, new_name + "." + image_ext)
                )
                or os.path.exists(
                    os.path.join(label_dir, new_name + ".txt")
                )
            ):
                new_name = str(uuid.uuid4()).replace("-", "")

            rename_mapping[base_name] = new_name
        else:
            print(
                f"Warning: {os.path.basename(image_file)} "
                f"has no corresponding .txt file"
            )

    for old_base, new_name in rename_mapping.items():
        old_image = os.path.join(image_dir, old_base + "." + image_ext)
        old_txt = os.path.join(label_dir, old_base + ".txt")
        new_image = os.path.join(image_dir, new_name + "." + image_ext)
        new_txt = os.path.join(label_dir, new_name + ".txt")

        try:
            os.rename(old_image, new_image)
            os.rename(old_txt, new_txt)
            renamed_count += 1
            print(f"Renamed: {old_base}.{image_ext} -> {new_name}.{image_ext}")
            print(f"Renamed: {old_base}.txt -> {new_name}.txt")
        except OSError as e:
            print(f"Error renaming {old_image} or {old_txt}: {e}")

    print("\nRenaming complete!")
    print(f"Renamed {renamed_count} file pairs out of {total_files} total files")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, "obj_train_data")

    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=None,
        help=(
            "Flat directory of image/.txt pairs, or a parent directory "
            f"that contains images/ and labels/ (default: {default_dir}). "
            "Ignored when DIRECTORY is also given."
        ),
    )
    parser.add_argument(
        "directory_pos",
        nargs="?",
        default=None,
        metavar="DIRECTORY",
        help=(
            "Dataset directory. Same role as --directory. "
            f"Used instead of --directory when both are given "
            f"(default: {default_dir})."
        ),
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["png", "jpg"],
        default="jpg",
        help=(
            "Image extension to rename: jpg or png "
            "(default: jpg). Matching .txt labels are renamed with them."
        ),
    )

    args = parser.parse_args()
    chosen_dir = args.directory_pos or args.directory or default_dir
    data_dir = os.path.abspath(chosen_dir)

    print(f"Renaming files with UUID4 in: {data_dir}")
    print(f"Image type: .{args.type}")
    print("=" * 50)

    response = input("This will permanently rename files. Continue? (y/N): ")
    if response.lower() in ["y", "yes"]:
        rename_files_with_uuid(data_dir, args.type)
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()
