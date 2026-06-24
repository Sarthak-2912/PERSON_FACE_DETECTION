import cv2
import time
import os
import psutil
from insightface.model_zoo import get_model

detector = get_model("models/det_500m.onnx")
detector.prepare(ctx_id=-1, input_size=(640, 640))

process = psutil.Process(os.getpid())
logical_cpus = psutil.cpu_count(logical=True)

last_cpu = process.cpu_times()
last_wall = time.perf_counter()

cap = cv2.VideoCapture(0)

prev_time = time.perf_counter()
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    start_inference = time.perf_counter()

    bboxes, kpss = detector.detect(frame)

    inference_time = (time.perf_counter() - start_inference) * 1000

    current_time = time.perf_counter()
    instant_fps = 1.0 / (current_time - prev_time)
    fps = instant_fps if fps == 0 else (0.9 * fps + 0.1 * instant_fps)
    prev_time = current_time

    current_cpu = process.cpu_times()
    current_wall = time.perf_counter()

    cpu_delta = (
        (current_cpu.user + current_cpu.system)
        - (last_cpu.user + last_cpu.system)
    )

    wall_delta = current_wall - last_wall

    cpu_usage = 0.0
    if wall_delta > 0:
        cpu_usage = (cpu_delta / wall_delta) * 100.0 / logical_cpus

    last_cpu = current_cpu
    last_wall = current_wall

    memory_mb = process.memory_info().rss / (1024 * 1024)

    face_count = len(bboxes)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4].astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {inference_time:.2f} ms",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"CPU: {cpu_usage:.1f}%",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"RAM: {memory_mb:.1f} MB",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()