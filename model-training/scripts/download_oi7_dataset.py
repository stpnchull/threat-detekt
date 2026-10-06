# Computer Vision Engineer
#
# This project incorporates components from the Apache 2.0 licensed project.
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

# ******************************************************************************
# DISCLAIMER:
#
# This script downloads images and bounding-box annotations from Google Open
# Images. Those files are covered by their own licenses. Read the terms at
# https://storage.googleapis.com/openimages/web/index.html and follow them
# for every image and annotation this script saves.
#
# By using this script, you agree to those terms. Use that breaks them is
# your responsibility.
# ******************************************************************************

"""Download Open Images boxes for the classes you name.

The pictures and labels are Google Open Images data. Read the license
terms at https://storage.googleapis.com/openimages/web/index.html
before you use them. This script needs the pandas
and requests packages, a network connection, and a python command on
your PATH.

    python download_oi7_dataset.py \\
        --classes "['Handgun', 'Rifle']" \\
        --out-dir ./data \\
        --max-number-images-per-class 1000 \\
        --yolov8-format True

Class names
--classes is a Python list of Open Images display names. Matching is
exact, including capitalization. Handgun matches. handgun does not.
The default list is person, handgun, rifle, shotgun, and knife. A name
that is not in the Open Images class table stops the script.

Each requested name becomes one YOLO class id, in the order you listed
them. The first name is class 0.

How many images
--max-number-images-per-class limits how many image ids are queued for
each class. The default is 1000. The annotation tables are scanned in
order: train, then validation, then test. An image id is queued once.
It counts toward the first requested class found on that image. Later
classes on the same image do not get their own copy of it.

The script still writes every requested box that belongs to a picture
that was actually downloaded.

What gets downloaded
On the first run the current directory receives the class table, the
train, validation, and test box tables, and the Open Images
downloader.py. Later runs reuse those files. The class table is the v7
file. The train boxes are the v6 file. The validation and test boxes
are the v5 files.

JPEGs are fetched into a temporary .out folder in the current directory.
That folder is removed at the end. A picture that fails to download
gets no label file.

Where files are saved
--out-dir defaults to ./data. The script deletes and recreates the
split folders it owns inside that directory. Other files there are
left in place.

With --yolov8-format True, which is the default, the layout is:

    data/
      images/train/
      images/val/
      images/test/
      labels/train/
      labels/val/
      labels/test/

The Open Images validation split is saved under val. Each label line is
YOLO format, with box values already between 0 and 1:

    class_id x_center y_center width height

Pass --yolov8-format False to keep the earlier layout instead:

    data/train/imgs/   data/train/anns/
    data/val/imgs/     data/val/anns/
    data/test/imgs/    data/test/anns/

The values that turn the YOLO layout on are True, T, and 1. Any other
value turns it off.

Examples
One class, up to 1000 images, YOLO folders:

    python download_oi7_dataset.py \\
        --classes "['Handgun']" \\
        --out-dir ./data/handgun \\
        --max-number-images-per-class 1000 \\
        --yolov8-format True

The default class list, written to ./data:

    python download_oi7_dataset.py
"""

import argparse
import ast
import os
import shutil
import sys

import pandas as pd
import requests


def process(classes, data_out_dir, yolov8_format, max_number_images_per_class):
    if max_number_images_per_class is None:
        max_number_images_per_class = sys.maxsize

    train_data_url = (
        "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    )
    val_data_url = (
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
    )
    test_data_url = (
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
    )

    downloader_url = (
        "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
    )

    class_names_all_url = (
        "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
    )

    for url in [
        train_data_url,
        val_data_url,
        test_data_url,
        class_names_all_url,
        downloader_url,
    ]:
        if not os.path.exists(url.split("/")[-1]):
            print("downloading {}...".format(url.split("/")[-1]))
            r = requests.get(url)
            with open(url.split("/")[-1], "wb") as f:
                f.write(r.content)

    class_ids = []

    classes_all = pd.read_csv(class_names_all_url.split("/")[-1])

    for class_ in classes:
        if class_ not in list(classes_all["DisplayName"]) or class_ not in list(
            classes_all["DisplayName"]
        ):
            raise Exception("Class name not found: {}".format(class_))
        class_index = list(classes_all["DisplayName"]).index(class_)
        class_ids.append(classes_all["LabelName"].iloc[class_index])

    image_list_file_path = os.path.join(".", "image_list_file")
    if os.path.exists(image_list_file_path):
        os.remove(image_list_file_path)

    image_list_file_list = []
    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        image_list_file_per_class = [[] for j in class_ids]
        filename = url.split("/")[-1]
        with open(filename, "r") as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(",")[
                    :13
                ]
                if (
                    class_name in class_ids
                    and id not in image_list_file_list
                    and len(image_list_file_per_class[class_ids.index(class_name)])
                    < max_number_images_per_class
                ):
                    image_list_file_list.append(id)
                    image_list_file_per_class[class_ids.index(class_name)].append(id)
                    with open(image_list_file_path, "a") as fw:
                        fw.write(
                            "{}/{}\n".format(["train", "validation", "test"][j], id)
                        )
                line = f.readline()

            f.close()

    out_dir = "./.out"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.system(
        "python downloader.py {} --download_folder={}".format(
            image_list_file_path, out_dir
        )
    )

    DATA_ALL_DIR = out_dir

    for set_ in ["train", "val", "test"]:
        for dir_ in [
            os.path.join(data_out_dir, set_),
            os.path.join(data_out_dir, set_, "imgs"),
            os.path.join(data_out_dir, set_, "anns"),
        ]:
            if os.path.exists(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_)

    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        filename = url.split("/")[-1]
        set_ = ["train", "val", "test"][j]
        print(filename)
        with open(filename, "r") as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(",")[
                    :13
                ]
                if class_name in class_ids:
                    if os.path.exists(os.path.join(DATA_ALL_DIR, "{}.jpg".format(id))):
                        if not os.path.exists(
                            os.path.join(
                                data_out_dir, set_, "imgs", "{}.jpg".format(id)
                            )
                        ):
                            shutil.copy(
                                os.path.join(DATA_ALL_DIR, "{}.jpg".format(id)),
                                os.path.join(
                                    data_out_dir, set_, "imgs", "{}.jpg".format(id)
                                ),
                            )
                        with open(
                            os.path.join(
                                data_out_dir, set_, "anns", "{}.txt".format(id)
                            ),
                            "a",
                        ) as f_ann:
                            # class_id x_center y_center width height
                            x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                            xc = (x1 + x2) / 2
                            yc = (y1 + y2) / 2
                            w = x2 - x1
                            h = y2 - y1

                            f_ann.write(
                                "{} {} {} {} {}\n".format(
                                    int(class_ids.index(class_name)), xc, yc, w, h
                                )
                            )
                            f_ann.close()

                line = f.readline()

    shutil.rmtree(out_dir, ignore_errors=True)

    if yolov8_format:
        for set_ in ["train", "val", "test"]:
            for dir_ in [
                os.path.join(data_out_dir, "images", set_),
                os.path.join(data_out_dir, "labels", set_),
            ]:
                if os.path.exists(dir_):
                    shutil.rmtree(dir_)
                os.makedirs(dir_)

            for filename in os.listdir(os.path.join(data_out_dir, set_, "imgs")):
                shutil.copy(
                    os.path.join(data_out_dir, set_, "imgs", filename),
                    os.path.join(data_out_dir, "images", set_, filename),
                )
            for filename in os.listdir(os.path.join(data_out_dir, set_, "anns")):
                shutil.copy(
                    os.path.join(data_out_dir, set_, "anns", filename),
                    os.path.join(data_out_dir, "labels", set_, filename),
                )

            shutil.rmtree(os.path.join(data_out_dir, set_))


if __name__ == "__main__":
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--classes",
        default=["person", "handgun", "rifle", "shotgun", "knife"],
        help=(
            "Python list of Open Images display names, for example "
            "\"['Handgun', 'Rifle']\". Names must match exactly, "
            "including capitalization. "
            "Default: person, handgun, rifle, shotgun, knife."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="./data",
        help="Directory for the images and labels (default: ./data).",
    )
    parser.add_argument(
        "--yolov8-format",
        default=True,
        help=(
            "True, T, or 1 writes images/<split> and labels/<split>. "
            "Any other value writes <split>/imgs and <split>/anns. "
            "Default: True."
        ),
    )
    parser.add_argument(
        "--max-number-images-per-class",
        default=1000,
        help=(
            "Maximum image ids to queue for each class (default: 1000). "
            "Each image id is queued once."
        ),
    )
    args = parser.parse_args()

    print("Args", args)

    classes = args.classes
    print("Classes", classes)
    if type(classes) is str:
        classes = ast.literal_eval(classes)

    out_dir = args.out_dir

    yolov8_format = True if args.yolov8_format in ["T", "True", 1, "1"] else False

    max_number_images_per_class = (
        int(args.max_number_images_per_class)
        if args.max_number_images_per_class is not None
        else None
    )

    process(classes, out_dir, yolov8_format, max_number_images_per_class)
