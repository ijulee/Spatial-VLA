from ultralytics import YOLO
import cv2
import os
import sys

def ensure_dir(path):
    """create a path if doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def process_image(model, image_path):
    """process single image"""
    results = model.predict(source=image_path)
    results[0].show()
    print("\nDetected objects:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class {cls} | Confidence {conf:.2f} | Box {xyxy}")

def process_video(model, video_path, output_root="output"):
    """process video: extract frame, detect, and save data"""
    frames_dir = os.path.join(output_root, "frames")
    labels_dir = os.path.join(output_root, "labels")
    ensure_dir(frames_dir)
    ensure_dir(labels_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"can't open: {video_path}")
        return

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"processed {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)

        # detect
        results = model.predict(source=frame, verbose=False)
        result = results[0]

        # save
        annotated = result.plot()  # show box in frames
        cv2.imwrite(frame_path, annotated)

        # save data
        label_path = os.path.join(labels_dir, f"frame_{frame_idx:05d}.txt")
        with open(label_path, "w") as f:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xywhn = box.xywhn[0].tolist()  
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in xywhn)} {conf:.4f}\n")

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    print(f"\nFinished! frames are saved to: {frames_dir}\nlables are saved to: {labels_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_and_extract.py <image_or_video_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    model = YOLO("yolo12s.pt")

    # print model type
    print("\nLoaded model classes:")
    print(model.names)

    # identify input type
    if input_path.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(model, input_path)
    elif input_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        process_video(model, input_path)
    else:
        print("invalid file format, please input image or video")

if __name__ == "__main__":
    main()