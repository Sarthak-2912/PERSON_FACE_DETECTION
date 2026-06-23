import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
from ultralytics import YOLO
import cv2
import time
import psutil
model = YOLO(
    "models/yolov8n-face-lindevs.onnx",
    task="detect"
)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
process = psutil.Process()
process.cpu_percent()
prev_time = time.time()
frame_count = 0
inference_ms = 0
trackers = []
DETECTION_INTERVAL = 30
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    frame_count += 1
    face_boxes = []
    if frame_count % DETECTION_INTERVAL == 0:
        start = time.time()
        results = model(
            frame,
            conf=0.6,
            imgsz=640,
            verbose=False
        )
        end = time.time()
        inference_ms = (end - start) * 1000
        trackers.clear()
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    face_boxes.append(
                        (x1, y1, x2, y2)
                    )
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker.init(
                        frame,
                        (
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1
                        )
                    )
                    trackers.append(tracker)
    else:
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                face_boxes.append(
                    (
                        x,
                        y,
                        x + w,
                        y + h
                    )
                )
    annotated_frame = frame.copy()
    for (x1, y1, x2, y2) in face_boxes:
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    process_cpu = process.cpu_percent(interval=None)

    cpu_count = psutil.cpu_count(logical=True) or 1

    model_cpu_usage = process_cpu / cpu_count

    memory_usage = (
        process.memory_info().rss
        / 1024
        / 1024
    )
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated_frame,
        f"Model CPU: {model_cpu_usage:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated_frame,
        f"Memory: {memory_usage:.1f} MB",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated_frame,
        f"Inference: {inference_ms:.1f} ms",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow(
        "YOLO Face Detection + MOSSE Tracker",
        annotated_frame
    )
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        print("Exiting...")
        break
cap.release()
cv2.destroyAllWindows()