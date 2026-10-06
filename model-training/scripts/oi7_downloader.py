# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download Open Images JPEGs from a list of image ids.

Give this script a text file of Open Images ids. It fetches each JPEG from
the public open-images-dataset bucket and saves it locally. No AWS
credentials are required. The boto3 and tqdm packages are required, plus
a network connection.

    python oi7_downloader.py image_list_file --download_folder ./.out

Image list
The list is a text file with one image per line:

    train/f9e0434389a1d4dd
    validation/1a007563ebc18664
    test/ea8bfd4e765304db

The split is train, validation, test, or challenge2018. The id is the
hexadecimal Open Images id. A trailing .jpg on the id is removed, so
train/f9e0434389a1d4dd.jpg is the same request. A line that does not
match this pattern stops the run before any download. The message
includes that line's number, counting from 0.

Where files are saved
Each file is written as <image_id>.jpg in one folder. The split is not
added to the file name or as a subfolder. --download_folder is created
when it is missing. When the flag is omitted, files are written in the
current directory.

Downloads run five at a time unless you pass --num_processes. A failure
to fetch one image stops the run and prints the split and id.

Examples
Save a prepared list into ./.out:

    python oi7_downloader.py image_list_file --download_folder ./.out

Use 10 downloads at a time, writing into the current directory:

    python oi7_downloader.py image_list_file --num_processes 10
"""

import argparse
import os
import re
import sys
from concurrent import futures

import boto3
import botocore
import tqdm

BUCKET_NAME = "open-images-dataset"
REGEX = r"(test|train|validation|challenge2018)/([a-fA-F0-9]*)"


def check_and_homogenize_one_image(image):
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f"ERROR in line {line_number} of the image list. The following image "
                f'string is not recognized: "{image}".'
            )


def read_image_list_file(image_list_file):
    with open(image_list_file, "r") as f:
        for line in f:
            yield line.strip().replace(".jpg", "")


def download_one_image(bucket, split, image_id, download_folder):
    try:
        bucket.download_file(
            f"{split}/{image_id}.jpg", os.path.join(download_folder, f"{image_id}.jpg")
        )
    except botocore.exceptions.ClientError as exception:
        sys.exit(f"ERROR when downloading image `{split}/{image_id}`: {str(exception)}")


def download_all_images(args):
    """Download every image in the list into download_folder.

    Missing folders are created. Files are named <image_id>.jpg.
    """
    bucket = boto3.resource(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    ).Bucket(BUCKET_NAME)

    download_folder = args["download_folder"] or os.getcwd()

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    try:
        image_list = list(
            check_and_homogenize_image_list(read_image_list_file(args["image_list"]))
        )
    except ValueError as exception:
        sys.exit(exception)

    progress_bar = tqdm.tqdm(
        total=len(image_list), desc="Downloading images", leave=True
    )
    with futures.ThreadPoolExecutor(max_workers=args["num_processes"]) as executor:
        all_futures = [
            executor.submit(
                download_one_image, bucket, split, image_id, download_folder
            )
            for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image_list",
        type=str,
        default=None,
        help=(
            "Text file of split/image_id lines. "
            "Split is train, validation, test, or challenge2018."
        ),
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=5,
        help="How many images to download at once (default: 5).",
    )
    parser.add_argument(
        "--download_folder",
        type=str,
        default=None,
        help=(
            "Folder for the JPEGs, named <image_id>.jpg. "
            "Created if missing. Default: the current directory."
        ),
    )
    download_all_images(vars(parser.parse_args()))
