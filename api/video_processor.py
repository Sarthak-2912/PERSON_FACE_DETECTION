import cv2
import base64
import time
import psutil
import os
import asyncio

from api.detector import detect_faces

DETECTION_QUEUE = asyncio.Queue()


def encode_frame(frame):

    success, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    if not success:
        return None

    return base64.b64encode(
        buffer
    ).decode("utf-8")


def draw_boxes(frame, faces):

    for bbox in faces:

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return frame


async def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("Unable to open video")

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    sample_interval = int(
        fps * 0.5
    )

    process = psutil.Process(
        os.getpid()
    )

    start_time = time.perf_counter()

    frame_number = 0

    samples_processed = 0
    anomalies_detected = 0

    total_inference_ms = 0

    print(
        f"Started processing: {video_path}"
    )

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret1, frame1 = cap.read()

        if not ret1:
            break

        ret2, frame2 = cap.read()

        if not ret2:
            break

        inference_start = (
            time.perf_counter()
        )

        faces1 = detect_faces(frame1)
        faces2 = detect_faces(frame2)

        inference_ms = (
            time.perf_counter()
            - inference_start
        ) * 1000

        total_inference_ms += (
            inference_ms
        )

        samples_processed += 1

        face_count_1 = len(faces1)
        face_count_2 = len(faces2)

        max_faces = max(
            face_count_1,
            face_count_2
        )

        print(
            f"Frame {frame_number} | "
            f"Faces={max_faces}"
        )

        # Send ONLY anomaly frames
        if max_faces > 1:

            frame1 = draw_boxes(
                frame1,
                faces1
            )

            frame2 = draw_boxes(
                frame2,
                faces2
            )

            timestamp = round(
                frame_number / fps,
                3
            )

            anomaly_event = {

                "event_type":
                    "MULTIPLE_PERSON_DETECTED",

                "timestamp":
                    timestamp,

                "frame_number_1":
                    frame_number,

                "frame_number_2":
                    frame_number + 1,

                "person_count":
                    max_faces,

                "frame1":
                    encode_frame(
                        frame1
                    ),

                "frame2":
                    encode_frame(
                        frame2
                    )
            }

            await DETECTION_QUEUE.put(
                anomaly_event
            )

            print(
                f"ANOMALY -> "
                f"{timestamp} sec | "
                f"Faces={max_faces}"
            )

            anomalies_detected += 1

            await asyncio.sleep(0)

        frame_number += (
            sample_interval
        )

    cap.release()

    processing_time = (
        time.perf_counter()
        - start_time
    )

    cpu_usage = (
        psutil.cpu_percent()
    )

    memory_mb = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    avg_inference = 0

    if samples_processed > 0:

        avg_inference = (
            total_inference_ms
            / samples_processed
        )

    result = {

        "status":
            "success",

        "video_fps":
            round(fps, 2),

        "samples_processed":
            samples_processed,

        "anomalies_detected":
            anomalies_detected,

        "average_inference_ms":
            round(
                avg_inference,
                2
            ),

        "cpu_usage_percent":
            cpu_usage,

        "memory_mb":
            round(
                memory_mb,
                2
            ),

        "processing_time_sec":
            round(
                processing_time,
                2
            )
    }

    print("\nProcessing completed")
    print(result)

    return result