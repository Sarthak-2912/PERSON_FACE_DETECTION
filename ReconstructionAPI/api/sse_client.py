import cv2
import base64
import numpy as np
import requests
import sseclient
import json
import time

DETECTION_API = "http://127.0.0.1:8000"


def decode_frame(encoded):

    image_bytes = base64.b64decode(
        encoded
    )

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    return cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )


def receive_frames(
    job_id
):

    received_frames = {}

    url = (

        DETECTION_API

        + "/stream-detections/"

        + job_id

    )

    print()

    print(
        f"[{job_id}] Connecting to Detection API..."
    )

    while True:

        try:

            response = requests.get(

                url,

                stream=True,

                timeout=30

            )

            if response.status_code == 200:
                break

        except requests.RequestException:
            pass

        print(
            f"[{job_id}] Waiting for Detection API..."
        )

        time.sleep(1)

    client = sseclient.SSEClient(
        response
    )

    for event in client.events():

        try:

            if not event.data:
                continue

            data = json.loads(
                event.data
            )

            if data.get("event_type") == "PROCESSING_COMPLETE":

                print()

                print(
                    f"[{job_id}] Detection Completed"
                )

                break

            frame1 = decode_frame(
                data["frame1"]
            )

            frame2 = decode_frame(
                data["frame2"]
            )

            received_frames[
                data["frame_number_1"]
            ] = frame1

            received_frames[
                data["frame_number_2"]
            ] = frame2

            print(

                f"[{job_id}] "

                f"Received Frames "

                f"{data['frame_number_1']} "

                f"& "

                f"{data['frame_number_2']}"

            )

        except Exception as e:

            print(
                f"[{job_id}] {e}"
            )

    return received_frames