from ultralytics import YOLO
import cv2
import time

model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture("videos/sample.mp4")
prev_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video Finished")
        break
    results = model(
        frame, 
        conf=0.25,
        verbose=False
    )
    annotated_frame = results[0].plot()
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("Video Detection", annotated_frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()