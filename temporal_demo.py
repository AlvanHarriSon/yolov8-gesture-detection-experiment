"""Convert frame-level detections into stable gesture events."""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


class DatasetStream:
    def __init__(self, directory, class_names, frames_per_class=40):
        groups = defaultdict(list)
        for path in sorted(directory.glob("*.jpg")):
            class_name = next((name for name in class_names if path.name.startswith(name + "_")), None)
            if class_name:
                groups[class_name].append(path)
        self.frames = []
        for class_name in sorted(groups):
            samples = groups[class_name][:8]
            self.frames.extend(samples[index % len(samples)] for index in range(frames_per_class))
        self.index = 0

    def isOpened(self):
        return bool(self.frames)

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = cv2.imread(str(self.frames[self.index]))
        self.index += 1
        return True, frame

    def release(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "runs" / "normal" / "weights" / "best.pt")
    parser.add_argument("--source", choices=["dataset", "camera"], default="dataset")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "images" / "val")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--cooldown-frames", type=int, default=40)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_size = model.overrides.get("imgsz", 320)
    class_names = list(model.names.values())
    stream = (cv2.VideoCapture(0) if args.source == "camera"
              else DatasetStream(args.dataset, class_names))
    if not stream.isOpened():
        raise SystemExit("Unable to open the selected input source.")

    recent, events = [], []
    cooldown = 0
    while True:
        ok, frame = stream.read()
        if not ok:
            break

        result = model(frame, conf=0.01, imgsz=image_size, verbose=False)[0]
        current, best_confidence = None, 0.0
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > best_confidence:
                best_confidence = confidence
                current = model.names[int(box.cls[0])]
        if best_confidence < args.confidence:
            current = None

        recent.append(current)
        recent = recent[-args.stable_frames:]
        fired = None
        if cooldown > 0:
            cooldown -= 1
        elif (len(recent) == args.stable_frames and recent[0] is not None
              and all(value == recent[0] for value in recent)):
            fired = recent[0]
            events.append(fired)
            recent.clear()
            cooldown = args.cooldown_frames
            print(f"event={fired} count={len(events)}")

        status = f"FIRED: {fired}" if fired else (f"seeing {current}" if current else "")
        cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (60, 180, 75), 2)
        for box in result.boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 75), 2)

        if not args.headless:
            cv2.imshow("gesture detection", frame)
            if cv2.waitKey(1 if args.source == "camera" else 60) & 0xFF == ord("q"):
                break

    stream.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"events={events}")


if __name__ == "__main__":
    main()
