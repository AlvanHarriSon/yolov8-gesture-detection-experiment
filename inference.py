"""Run single-image gesture detection and save an annotated result."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "gestures.pt")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=ROOT / "inference_result.jpg")
    args = parser.parse_args()

    image_path = args.image
    if image_path is None:
        candidates = sorted((ROOT / "raw" / "images").glob("*.jpg"))
        if not candidates:
            raise SystemExit("No input image found. Pass --image or populate raw/images/.")
        image_path = candidates[0]

    model = YOLO(args.weights)
    result = model(str(image_path), conf=args.conf, verbose=False)[0]
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Unable to read image: {image_path}")

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        label = model.names[class_id]
        detections.append((label, confidence, [x1, y1, x2, y2]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (60, 180, 75), 3)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 180, 75), 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise SystemExit(f"Unable to write output: {args.output}")

    print(f"image={image_path}")
    print(f"detections={len(detections)}")
    for label, confidence, coordinates in detections:
        print(f"{label:<12} confidence={confidence:.3f} box={coordinates}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
