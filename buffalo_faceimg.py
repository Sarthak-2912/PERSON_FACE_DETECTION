import cv2
import time
import os
import psutil
from insightface.model_zoo import get_model

detector = get_model(
    "models/det_500m.onnx",
    providers=["CPUExecutionProvider"]
)
detector.prepare(ctx_id=-1, input_size=(640, 640))

process = psutil.Process(os.getpid())

image_path = "images/test.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

start_time = time.perf_counter()

bboxes, kpss = detector.detect(image)

inference_time = (time.perf_counter() - start_time) * 1000

memory_mb = process.memory_info().rss / (1024 * 1024)

face_count = len(bboxes)

for bbox in bboxes:
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

cv2.putText(
    image,
    f"Faces: {face_count}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"Inference: {inference_time:.2f} ms",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"RAM: {memory_mb:.1f} MB",
    (10, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imshow("SCRFD det_500m Face Detection", image)

cv2.imwrite("outputs/output(buffalo-s img).jpg", image)

print(f"Faces Detected : {face_count}")
print(f"Inference Time : {inference_time:.2f} ms")
print(f"RAM Usage      : {memory_mb:.1f} MB")
print("Output Saved   : outputs/output(buffalo-s img).jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()