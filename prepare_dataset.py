"""Create a YOLO dataset with model-assisted boxes and person-disjoint splits."""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
CLASSES = ["like", "stop", "fist", "hand_heart", "timeout", "xsign"]


def gesture_from_filename(filename):
    return next((name for name in CLASSES if filename.startswith(name + "_")), None)


def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    left, top = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
    right, bottom = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
    return (left + right) / 2, (top + bottom) / 2, right - left, bottom - top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=ROOT / "raw" / "images")
    parser.add_argument("--metadata", type=Path, default=ROOT / "raw" / "meta.json")
    parser.add_argument("--region-weights", type=Path, default=ROOT / "weights" / "region.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    people = sorted(set(metadata.values()))
    random.Random(args.seed).shuffle(people)
    val_count = max(1, int(len(people) * args.val_fraction))
    val_people = set(people[:val_count])

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    region_model = YOLO(args.region_weights)
    counts = {name: {"train": 0, "val": 0} for name in CLASSES}
    dropped = 0

    for image_path in sorted(args.images.glob("*.jpg")):
        gesture = gesture_from_filename(image_path.name)
        if gesture is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            dropped += 1
            continue
        result = region_model(image, conf=0.25, imgsz=640, verbose=False)[0]
        if len(result.boxes) == 0:
            dropped += 1
            continue

        best = int(result.boxes.conf.argmax())
        box = [float(value) for value in result.boxes.xyxy[best].tolist()]
        height, width = image.shape[:2]
        cx, cy, bw, bh = normalize_box(box, width, height)
        split = "val" if metadata.get(image_path.name) in val_people else "train"
        class_id = CLASSES.index(gesture)

        shutil.copy2(image_path, args.output / "images" / split / image_path.name)
        label_path = args.output / "labels" / split / f"{image_path.stem}.txt"
        label_path.write_text(
            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
        )
        counts[gesture][split] += 1

    yaml_lines = [
        f"path: {args.output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
    ]
    (args.output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    train_people = {
        metadata[path.name] for path in (args.output / "images" / "train").glob("*.jpg")
    }
    actual_val_people = {
        metadata[path.name] for path in (args.output / "images" / "val").glob("*.jpg")
    }
    overlap = train_people & actual_val_people

    print(f"people={len(people)} val_people={len(val_people)} dropped={dropped}")
    for name, split_counts in counts.items():
        print(f"{name:<12} train={split_counts['train']:>3} val={split_counts['val']:>3}")
    print(f"person_overlap={len(overlap)}")
    if overlap:
        raise SystemExit("Person leakage detected between train and validation splits.")


if __name__ == "__main__":
    main()
