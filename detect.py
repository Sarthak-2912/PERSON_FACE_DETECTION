from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")

image = cv2.imread("images/test.jpg")

results = model(
    image,
    imgsz=1280,
    conf=0.25
)

annotated_image = results[0].plot()

cv2.imwrite(
    "outputs/result.jpg",
    annotated_image
)

print("Detection Complete!")