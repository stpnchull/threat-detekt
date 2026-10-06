#!/usr/bin/env python3
"""Find identical images in a directory.

Point this script at a folder of images. It hashes each image and groups
files whose contents are the same, even when the file names differ.
Only files sitting directly in that folder are checked.

    python find_duplicates.py ./datasets/images

Pass the folder as a positional argument or with --directory. When both
are given, the positional path is used.

Which files
The comparison includes these image extensions, in any capitalization:

    .jpg .jpeg .jpe .jfif .jif
    .png .gif .bmp .dib
    .tif .tiff .tga
    .webp .avif
    .heic .heif
    .ico
    .jp2 .j2k .jpf .jpx
    .pbm .pgm .ppm .pnm
    .svg

Other files in the folder, including .txt labels, are left alone.

How duplicates are decided
Two images match when their bytes are identical. The hash is MD5.
A progress line prints every 100 files.

Each group lists the file name and size. The summary counts extra
copies and the space those copies use. The first file in alphabetical
order is treated as the one to keep.

Deleting copies
When duplicates are found, the script asks before it deletes anything.
Answer y or yes to delete every file in a group except the one that
is kept. Any other answer leaves the files alone.

Examples
A folder of images:

    python find_duplicates.py ./datasets/images

The same folder, passed with the flag:

    python find_duplicates.py --directory ./datasets/images
"""

import argparse
import hashlib
import os
from collections import defaultdict


# Still-image extensions, compared case-insensitively.
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".dib",
    ".gif",
    ".heic",
    ".heif",
    ".ico",
    ".j2k",
    ".jfif",
    ".jif",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpf",
    ".jpg",
    ".jpx",
    ".pbm",
    ".pgm",
    ".png",
    ".pnm",
    ".ppm",
    ".svg",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
}


def calculate_md5(file_path, chunk_size=8192):
    """
    Calculate MD5 hash of a file.

    Args:
        file_path (str): Path to the file
        chunk_size (int): Size of chunks to read at a time (for large files)

    Returns:
        str: MD5 hash of the file contents
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (IOError, OSError) as e:
        print(f"Error reading {file_path}: {e}")
        return None


def find_duplicates(directory, file_extensions=None):
    """
    Find duplicate files in a directory based on MD5 hash.

    Args:
        directory (str): Directory to search for duplicates
        file_extensions (list): Extensions to check, including the dot.
            Defaults to IMAGE_EXTENSIONS. Matching ignores capitalization.

    Returns:
        dict: Dictionary with MD5 hash as key and list of file paths as values
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} does not exist!")
        return {}

    if file_extensions is None:
        extensions = IMAGE_EXTENSIONS
    else:
        extensions = {ext.lower() for ext in file_extensions}

    all_files = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in extensions
        and os.path.isfile(os.path.join(directory, name))
    )

    print(f"Found {len(all_files)} files to check for duplicates...")

    # Dictionary to store hash -> list of file paths
    hash_to_files = defaultdict(list)

    # Calculate MD5 hash for each file
    for i, file_path in enumerate(all_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing file {i + 1}/{len(all_files)}...")

        file_hash = calculate_md5(file_path)
        if file_hash:
            hash_to_files[file_hash].append(file_path)

    # Filter out files with unique hashes (no duplicates)
    duplicates = {
        hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1
    }

    return duplicates


def report_duplicates(duplicates):
    """
    Print a report of duplicate files.

    Args:
        duplicates (dict): Dictionary of duplicate files from find_duplicates()
    """
    if not duplicates:
        print("No duplicate files found!")
        return

    print(f"\nFound {len(duplicates)} groups of duplicate files:")
    print("=" * 60)

    total_duplicate_files = 0
    total_wasted_space = 0

    for i, (file_hash, file_paths) in enumerate(duplicates.items(), 1):
        print(f"\nGroup {i} (MD5: {file_hash[:16]}...):")
        print(f"  {len(file_paths)} duplicate files:")

        # Get file sizes for space calculation
        file_sizes = []
        for file_path in file_paths:
            try:
                size = os.path.getsize(file_path)
                file_sizes.append(size)
                print(f"    - {os.path.basename(file_path)} ({size:,} bytes)")
            except OSError:
                print(f"    - {os.path.basename(file_path)} (size unknown)")

        # Calculate wasted space (all but one file are duplicates)
        if file_sizes and len(file_sizes) > 1:
            wasted_space = sum(file_sizes[1:])  # All but the first file
            total_wasted_space += wasted_space
            print(
                f"    Wasted space: {wasted_space:,} bytes ({wasted_space / 1024 / 1024:.2f} MB)"
            )

        total_duplicate_files += len(file_paths) - 1  # -1 because we keep one original

    print("\nSummary:")
    print(f"  Total duplicate files: {total_duplicate_files}")
    print(
        f"  Total wasted space: {total_wasted_space:,} bytes ({total_wasted_space / 1024 / 1024:.2f} MB)"
    )


def main():
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
            "Folder of images to check. "
            "Ignored when DIRECTORY is also given."
        ),
    )
    parser.add_argument(
        "directory_pos",
        nargs="?",
        default=None,
        metavar="DIRECTORY",
        help=(
            "Folder of images to check. Same role as --directory. "
            "Used instead of --directory when both are given."
        ),
    )

    args = parser.parse_args()
    chosen_dir = args.directory_pos or args.directory
    if not chosen_dir:
        parser.error(
            "Pass the folder to search, either as DIRECTORY or with --directory."
        )

    directory = os.path.abspath(chosen_dir)

    print(f"Finding duplicate files in: {directory}")
    print(
        "Checking extensions: "
        + ", ".join(sorted(IMAGE_EXTENSIONS))
    )
    print("=" * 60)

    if not os.path.isdir(directory):
        print(f"Error: Directory does not exist: {directory}")
        return

    # Find duplicates
    duplicates = find_duplicates(directory)

    # Report results
    report_duplicates(duplicates)

    # Optional: Ask if user wants to delete duplicates
    if duplicates:
        response = input("\nWould you like to delete duplicate files? (y/N): ")
        if response.lower() in ["y", "yes"]:
            delete_duplicates(duplicates)


def delete_duplicates(duplicates):
    """
    Delete duplicate files, keeping only the first file in each group.

    Args:
        duplicates (dict): Dictionary of duplicate files from find_duplicates()
    """
    deleted_count = 0

    for file_hash, file_paths in duplicates.items():
        # Keep the first file, delete the rest
        keep_file = file_paths[0]
        delete_files = file_paths[1:]

        print(f"\nKeeping: {os.path.basename(keep_file)}")

        for file_path in delete_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")

    print(f"\nDeleted {deleted_count} duplicate files.")


if __name__ == "__main__":
    main()
