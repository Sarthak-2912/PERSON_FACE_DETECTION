import cv2
import requests
import os

DETECTION_API = "http://127.0.0.1:8000/process-frame-pair"

RECONSTRUCTION_API = "http://127.0.0.1:8001/upload-video"

FINISH_API = "http://127.0.0.1:8000/finish-detection"


def send_frame_pair(
    frame1,
    frame2,
    job_id,
    frame_number_1,
    frame_number_2
):

    success1, buffer1 = cv2.imencode(".jpg", frame1)
    success2, buffer2 = cv2.imencode(".jpg", frame2)

    if not success1 or not success2:
        return

    files = {
        "frame1": (
            "frame1.jpg",
            buffer1.tobytes(),
            "image/jpeg"
        ),
        "frame2": (
            "frame2.jpg",
            buffer2.tobytes(),
            "image/jpeg"
        )
    }

    data = {
        "job_id": job_id,
        "frame_number_1": frame_number_1,
        "frame_number_2": frame_number_2
    }

    try:

        response = requests.post(
            DETECTION_API,
            files=files,
            data=data,
            timeout=10
        )

        if response.status_code == 200:

            print(
                f"[{job_id}] Sent Frame Pair "
                f"({frame_number_1}, {frame_number_2})"
            )

        else:

            print(
                f"[{job_id}] Frame Pair Failed"
            )

            print(response.text)

    except Exception as e:

        print(f"[{job_id}] {e}")


def dispatch_reconstruction(
    video_path,
    job_id
):

    try:

        with open(video_path, "rb") as video:

            files = {
                "video": (
                    os.path.basename(video_path),
                    video,
                    "video/mp4"
                )
            }

            data = {
                "job_id": job_id
            }

            response = requests.post(
                RECONSTRUCTION_API,
                files=files,
                data=data,
                timeout=300
            )

        print()
        print("=================================")
        print("Video sent to Reconstruction API")
        print("=================================")
        print(response.status_code)
        print(response.text)

    except Exception as e:

        print(f"[{job_id}] Reconstruction Error : {e}")


def finish_detection(job_id):

    try:

        response = requests.post(

            FINISH_API,

            data={
                "job_id": job_id
            },

            timeout=10

        )

        print(
            f"[{job_id}] Detection Finished"
        )

        print(response.status_code)

    except Exception as e:

        print(
            f"[{job_id}] Finish Detection Error : {e}"
        )