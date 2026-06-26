import cv2
import base64
import asyncio
import requests
from api.video_processor import EXECUTOR

from api.detector import detect_faces
from api.video_processor import JOB_QUEUES

RECONSTRUCTION_API = "http://127.0.0.1:8001/upload-frame"




def encode_frame(frame):

    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


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


def send_to_reconstruction(
    frame,
    job_id,
    frame_number,
    timestamp
):

    success, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    if not success:
        return

    files = {

        "frame": (

            "frame.jpg",

            buffer.tobytes(),

            "image/jpeg"

        )

    }

    data = {

        "job_id": job_id,

        "frame_number": frame_number,

        "timestamp": timestamp

    }

    try:

        requests.post(

            RECONSTRUCTION_API,

            files=files,

            data=data,

            timeout=10

        )

    except Exception as e:

        print(
            f"[{job_id}] Reconstruction Upload Failed : {e}"
        )


async def process_frame(

    frame,

    job_id,

    frame_number,

    timestamp

):

    loop = asyncio.get_running_loop()

    faces = await loop.run_in_executor(

        EXECUTOR,

        detect_faces,

        frame

    )

    face_count = len(faces)

    print(

        f"[{job_id}] "

        f"Frame {frame_number}"

        f" Faces={face_count}"

    )

    if face_count <= 1:

        return

    frame = draw_boxes(

        frame,

        faces

    )

    send_to_reconstruction(

        frame,

        job_id,

        frame_number,

        timestamp

    )

    if job_id in JOB_QUEUES:

        await JOB_QUEUES[job_id].put(

            {

                "event_type": "MULTIPLE_PERSON_DETECTED",

                "job_id": job_id,

                "frame_number": frame_number,

                "timestamp": timestamp,

                "person_count": face_count,

                "frame": encode_frame(frame)

            }

        )

    print(

        f"[{job_id}] "

        f"Anomaly Sent "

        f"(Frame {frame_number})"

    )