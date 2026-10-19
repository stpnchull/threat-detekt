# Building a Custom Threat Detekt Dataset

This guide covers how to assemble a training dataset for person and weapon detection: starting from Google Open Images, correcting labels by hand, splitting train / val / test, adding hard negatives, and later growing the set with auto-labeling.

Open Images is a strong starting point. It is not a finished dataset for this problem. Plan on reviewing and correcting labels before you trust a training run.

## What you need

- Python environment with the packages in `model-training/requirements.txt` (`make deps` from `model-training/`)
- Network access to download Open Images files
- A labeling tool for review: [CVAT](https://www.cvat.ai/) or [Label Studio](https://labelstud.io/)
- Optional later: a trained checkpoint, video sources, and other public weapon-detection datasets

Target classes for this project:


| Class id | Name    |
| -------- | ------- |
| 0        | person  |
| 1        | handgun |
| 2        | rifle   |
| 3        | shotgun |
| 4        | knife   |




## Scripts involved


| Script                                           | Role                                                                                                                        |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `model-training/scripts/download_oi7_dataset.py` | Selects classes from Open Images, builds the image list, downloads JPEGs, writes YOLO labels and train / val / test folders |
| `model-training/scripts/oi7_downloader.py`       | Low-level JPEG fetcher. Reads a `split/image_id` list and pulls files from the public Open Images bucket                    |
| `model-training/scripts/yolo_split_report.py`    | Reports current split sizes and per-class balance                                                                           |
| `model-training/scripts/auto_label.py`           | Pseudo-labels new images or video frames once a model is reliable enough                                                    |


`download_oi7_dataset.py` calls an Open Images `downloader.py` during the run. The checked-in `oi7_downloader.py` is the same kind of tool: use it when you want to download a custom image list yourself (for example hard negatives) without re-running the full class export.

Open Images content has its own license. Read the terms at [Open Images](https://storage.googleapis.com/openimages/web/index.html) before you use the data.

## Step 1: Download positive classes from Open Images

From `model-training/scripts/` (or pass absolute paths):

```bash
python download_oi7_dataset.py \
  --classes "['Person', 'Handgun', 'Rifle', 'Shotgun', 'Knife']" \
  --out-dir ./data/oi7_weapons \
  --max-number-images-per-class 1000 \
  --yolov8-format True
```

Important details:

- Class names must match Open Images **exactly**, including capitalization. `Handgun` works; `handgun` does not.
- List order becomes YOLO class ids (first name → class `0`).
- `--max-number-images-per-class` caps how many image ids are queued per class (default `1000`). Each image id is queued once.
- Open Images already defines train / validation / test. Those become `train`, `val`, and `test` under your output directory.

Default YOLO layout:

```text
data/oi7_weapons/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

Each label line is:

```text
class_id x_center y_center width height
```

Box values are already normalized to `[0, 1]`.

On the first run, annotation CSVs and a `downloader.py` are saved in the current working directory and reused later. JPEGs are fetched into a temporary `.out` folder that is deleted when the script finishes.

### Using `oi7_downloader.py` directly

If you already have a list of Open Images ids (one per line as `train/<id>`, `validation/<id>`, or `test/<id>`):

```bash
python oi7_downloader.py image_list_file --download_folder ./.out --num_processes 5
```

Files are saved as `<image_id>.jpg` in one folder. You still need to create matching YOLO label files yourself (or leave them empty for negatives).

## Step 2: Do not trust Open Images labels as-is

Open Images boxes are large-scale and useful, but they are often incomplete or imprecise for weapon detection:

- Weapons can be missed, loosely boxed, or labeled under a related class.
- Occlusion, small objects, and unusual poses are common failure modes.
- Labels that are “good enough” for a general dataset are often not good enough for a security detector.

Treat the download as a **candidate set**. Import it into CVAT or Label Studio and:

1. Fix missing or wrong boxes.
2. Align class names with the Threat Detekt taxonomy above.
3. Drop images that are too ambiguous or out of scope.
4. Keep a clean held-out val / test set that you never auto-label later.

Human review at this stage matters more than collecting a larger unreviewed dump.

## Step 3: Configure train / val / test

Open Images already ships a split. That is a reasonable default. For a custom set you should still check that:

- Roughly **70% train / 15% val / 15% test** (adjust to your size; keep val and test large enough to be meaningful).
- Every class appears in val and test, not only in train.
- Images from the same video, scene, or near-duplicate group do not leak across splits.

Inspect the current state:

```bash
python yolo_split_report.py ./path/to/data.yaml
```

Point your YOLO `data.yaml` at the dataset root and the three image folders (see `model-training/config.yaml` for the field layout).

If the Open Images split is unbalanced for your classes, move images (and their matching label files) between `images/{train,val,test}` and `labels/{train,val,test}` until the report looks sane. Prefer moving whole images, not individual boxes.

## Step 4: Add hard negatives from Open Images

Hard negatives are images that look like they could confuse the model (tools, toys, phones, cables, long metal objects, people with no weapons) but must **not** be labeled as weapons.

How to build them with these scripts:

1. Choose Open Images classes that often confuse weapon detectors (examples: tools, toys, remote controls, musical instruments—pick names that exist in the Open Images class table).
2. Download those images with `download_oi7_dataset.py` into a separate folder, **or** build a custom `image_list_file` and run `oi7_downloader.py`.
3. For true negatives, give each image an **empty** `.txt` label (no boxes), or omit weapon labels after human review.
4. Distribute negatives into train / val / test with the same discipline as positives. Put some hard negatives in val and test so metrics reflect false-positive risk.

Without hard negatives, models often fire on everyday objects that share shape or context with weapons. Include them early.

## Step 5: Expand after the model is good enough

Once a checkpoint is accurate enough on a **human-labeled** val set, use `auto_label.py` to grow the training set from new imagery. Do not skip review forever; sample and audit auto-labels regularly.

```bash
python auto_label.py \
  --model ./weights/best.pt \
  --source ./unlabeled \
  --output ./datasets/auto_labeled \
  --frame-stride 30
```

Behavior that matters in practice:

- Only frames with high-confidence detections and no uncertain detections are saved.
- Skipped frames are **not** written as empty-label negatives.
- Tune `CLASS_THRESHOLDS` in `auto_label.py` against your labeled val set before trusting the output.
- Merge accepted pairs into train after a human spot-check; keep val / test human-labeled.



### Where to get more unlabeled data

Mix sources so the model sees more than Open Images stock photos:

- **YouTube (or similar) downloaders** — pull publicly available videos of ranges, security footage demos, news clips, or training material you are allowed to use. Run `auto_label.py` with `--frame-stride` so you do not keep every frame.
- **Other public weapon-detection datasets** — combine with your taxonomy carefully; remap class ids and re-check boxes.
- **Scene diversity** — change lighting (day / night / indoor / outdoor), camera distance, occlusion, weather, crowded scenes, and handheld vs fixed cameras. Accuracy gains here usually beat simply downloading more near-duplicates.

Also useful in this repo: `get_simple_image_samples.py` for quick keyword image pulls, `find_duplicates.py` before merging, and `rename_with_uuid.py` if you need unique file names across sources.

## Practical workflow

```text
Open Images download
        ↓
Human label review (CVAT / Label Studio)
        ↓
Add hard negatives + fix split (yolo_split_report)
        ↓
Train / evaluate
        ↓
Collect diverse video & images
        ↓
auto_label → spot-check → merge into train
        ↓
Retrain
```

Start smaller and cleaner rather than larger and noisy. A few thousand carefully reviewed images with hard negatives and a leak-free split will outperform a large dump of unchecked Open Images boxes for this task.