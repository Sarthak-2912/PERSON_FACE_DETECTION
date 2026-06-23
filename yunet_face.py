import cv2
import time
import psutil
model_path = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (640, 480)
)
cap = cv2.VideoCapture(0)
process = psutil.Process()
process.cpu_percent()
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    start = time.time()
    _, faces = detector.detect(frame)
    end = time.time()
    inference_ms = (end - start) * 1000
    if faces is not None:
        for face in faces:
            x, y, width, height = face[:4].astype(int)
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    process_cpu = process.cpu_percent()
    cpu_count = psutil.cpu_count(logical=True) or 1
    normalized_cpu = process_cpu / cpu_count
    memory_usage = (
        process.memory_info().rss
        / 1024
        / 1024
    )
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"CPU: {normalized_cpu:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"Memory: {memory_usage:.1f} MB",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"Inference: {inference_ms:.1f} ms",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow(
        "YuNet Face Detection",
        frame
    )
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()