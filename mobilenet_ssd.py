import cv2
import time
import psutil

prototxt = "models/MobileNetSSD_deploy.prototxt"
model = "models/Mobilenet_iter_73000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

cap = cv2.VideoCapture(0)

process = psutil.Process()
process.cpu_percent()

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    start = time.time()

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)

    detections = net.forward()

    end = time.time()

    inference_ms = (end - start) * 1000

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            startX, startY, endX, endY = box.astype("int")

            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                (0, 255, 0),
                2
            )

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    process_cpu = process.cpu_percent()

    cpu_count = psutil.cpu_count(logical=True) or 1

    normalized_cpu = process_cpu / cpu_count

    memory_usage = process.memory_info().rss / 1024 / 1024

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"CPU: {normalized_cpu:.1f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Memory: {memory_usage:.1f} MB", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Inference: {inference_ms:.1f} ms", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("MobileNet SSD Person Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()