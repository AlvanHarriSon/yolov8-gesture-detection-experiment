"""Train and validate the six-class YOLOv8 gesture detector."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--name", default="normal")
    args = parser.parse_args()

    start = "yolov8n.yaml" if args.from_scratch else ROOT / "weights" / "yolov8n.pt"
    model = YOLO(start)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, project=str(ROOT / "runs"),
        name=args.name, exist_ok=True,
    )
    metrics = model.val(
        data=str(args.data), imgsz=args.imgsz, device=args.device, workers=0
    )
    print(f"mAP@0.5={float(metrics.box.map50):.3f}")
    print(f"mAP@0.5:0.95={float(metrics.box.map):.3f}")
    for index, class_id in enumerate(metrics.box.ap_class_index):
        print(f"{model.names[int(class_id)]:<12} AP@0.5={float(metrics.box.ap50[index]):.3f}")
    print(f"best_weights={ROOT / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
