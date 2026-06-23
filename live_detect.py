from ultralytics import YOLO
import cv2
import time
import psutil
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(0)
process = psutil.Process()
process.cpu_percent()
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    start = time.time()
    results = model(
        frame,
        classes=[0],
        conf=0.25,
        imgsz=640,
        verbose=False
    )
    end = time.time()
    inference_ms = (end - start) * 1000
    annotated_frame = results[0].plot()
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cpu_usage = process.cpu_percent()
    normalized_cpu = cpu_usage / psutil.cpu_count()
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
        f"Process CPU: {normalized_cpu:.1f}%",
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
    cv2.imshow("Live Person Detection", annotated_frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        print("Exiting...")
        break
cap.release()
cv2.destroyAllWindows()