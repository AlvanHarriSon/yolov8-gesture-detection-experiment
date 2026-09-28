"""Evaluate confidence thresholds with class-aware IoU matching."""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def read_labels(path, width, height):
    labels = []
    for line in path.read_text().splitlines():
        class_id, cx, cy, bw, bh = line.split()
        cx, bw = float(cx) * width, float(bw) * width
        cy, bh = float(cy) * height, float(bh) * height
        labels.append({
            "class_id": int(class_id),
            "box": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
        })
    return labels


def match_counts(all_predictions, all_labels, threshold, iou_threshold=0.5):
    true_positive = false_positive = false_negative = 0
    for predictions, labels in zip(all_predictions, all_labels):
        candidates = [item for item in predictions if item["confidence"] >= threshold]
        claimed = [False] * len(labels)
        for prediction in candidates:
            best_index, best_iou = -1, 0.0
            for index, label in enumerate(labels):
                if claimed[index] or prediction["class_id"] != label["class_id"]:
                    continue
                overlap = iou(prediction["box"], label["box"])
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= iou_threshold:
                true_positive += 1
                claimed[best_index] = True
            else:
                false_positive += 1
        false_negative += sum(not value for value in claimed)
    return true_positive, false_positive, false_negative


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "runs" / "normal" / "weights" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "pr_curve.png")
    args = parser.parse_args()

    image_dir = args.dataset / "images" / "val"
    label_dir = args.dataset / "labels" / "val"
    model = YOLO(args.weights)
    image_size = model.overrides.get("imgsz", 320)
    all_predictions, all_labels = [], []

    for image_path in sorted(image_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        result = model(image, conf=0.01, imgsz=image_size, verbose=False)[0]
        predictions = [{
            "class_id": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "box": [float(value) for value in box.xyxy[0].tolist()],
        } for box in result.boxes]
        predictions.sort(key=lambda item: -item["confidence"])
        all_predictions.append(predictions)
        all_labels.append(read_labels(label_dir / f"{image_path.stem}.txt", width, height))

    records = []
    print(f"{'threshold':>9} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    for threshold in THRESHOLDS:
        tp, fp, fn = match_counts(all_predictions, all_labels, threshold)
        precision, recall, f1 = metrics(tp, fp, fn)
        records.append((f1, threshold, precision, recall))
        print(f"{threshold:>9.2f} {tp:>4} {fp:>4} {fn:>4} "
              f"{precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")

    best = max(records)
    plt.figure(figsize=(7.5, 6.2))
    plt.plot([item[3] for item in records], [item[2] for item in records], marker="o")
    for _, threshold, precision, recall in records:
        plt.annotate(f"{threshold:.2f}", (recall, precision), fontsize=8,
                     textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve by confidence threshold")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"best_threshold={best[1]:.2f} precision={best[2]:.3f} "
          f"recall={best[3]:.3f} f1={best[0]:.3f}")


if __name__ == "__main__":
    main()
