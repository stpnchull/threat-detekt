#!/usr/bin/env python3
"""Train the Threat Detekt detector from COCO-pretrained YOLOv8m weights.

Fill in config.yaml, then run this script from the model-training
directory. It trains for up to 150 epochs and writes the run to
runs/detect/train13/.

    python train.py

Requires the ultralytics and torch packages. Training is set up for a
CUDA GPU.

Dataset
Edit config.yaml in this directory before training. path is the dataset
root. train, val, and test are image folders relative to that root.
The class names are already listed there: person, handgun, rifle,
shotgun, and knife.

The script reads config.yaml from the current working directory, and
the run folder is created there as well.

What this run does
Unused GPU memory is cleared first. The starting weights are
yolov8m.pt, which were pretrained on COCO. Results go to
runs/detect/train13/. A later run writes into that same folder.

Training stops after 150 epochs, or earlier when validation shows no
improvement for 20 epochs. Images are resized to 640 pixels. Each step
uses 4 images, and gradients accumulate until the effective batch size
is 64. The full training split is used. Images are cached in memory,
with 8 loader workers, and training runs in mixed precision.

The best and last checkpoints are saved, and another checkpoint is
written every 25 epochs. Plots are saved in the same run folder.

Augmentation
Images may be rotated by up to 10 degrees. Mosaic is used until the
final 10 epochs. A horizontal flip is applied half the time. Mixup
runs on about 10% of samples and copy-paste on about 30%. Vertical
flip and shear are off. A small perspective warp is applied.

The learning rate starts at 0.01 and finishes at one tenth of that.
The first 3 epochs are warmup.

Change the values in this file to adjust the run. The run name is
train13.

Examples
From the model-training directory, after config.yaml is filled in:

    python train.py
"""

import argparse

import torch
from ultralytics import YOLO


def main() -> None:
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()

    torch.cuda.empty_cache()  # clear unused GPU memory before the run

    # model = YOLO("yolov8m.yaml")  # architecture only, no pretrained weights
    model = YOLO("yolov8m.pt")  # pretrained on COCO

    results = model.train(
        data="config.yaml",
        epochs=150,
        patience=20,  # stop when validation shows no improvement for 20 epochs
        batch=4,
        nbs=64,  # accumulate gradients to an effective batch size of 64
        imgsz=640,
        workers=8,
        cache=True,
        project="runs/detect",
        name="train13",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=25,
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,  # rotate up to 10 degrees for camera angles
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,  # slight perspective warp
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        box=7.5,
        cls=0.7,
        dfl=1.5,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
    )
    return results


if __name__ == "__main__":
    main()
