import cv2
import base64
import numpy as np
import requests
import sseclient
import json
import threading
import os
import time
import sys


SERVER = "http://127.0.0.1:8000"



if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = "uploads/video1.mp4"

OUTPUT_DIR = "outputs"

received_frames = {}

processing_finished = False

video_path_from_server = None

job_id = None


def decode_frame(encoded):

    image_bytes = base64.b64decode(
        encoded
    )

    array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )


def upload_video():

    global job_id

    print()

    print("Uploading Video...")

    with open(
        VIDEO_PATH,
        "rb"
    ) as f:

        files = {

            "video": (
                os.path.basename(
                    VIDEO_PATH
                ),
                f,
                "video/mp4"
            )

        }

        response = requests.post(

            SERVER + "/process-video",

            files=files

        )

    if response.status_code != 200:

        raise Exception(
            response.text
        )

    result = response.json()

    job_id = result["job_id"]

    print()

    print(
        f"Job ID : {job_id}"
    )

    return job_id


def listen_sse():

    global processing_finished
    global video_path_from_server

    url = (

        SERVER

        + "/stream-detections/"

        + job_id

    )

    print()

    print(

        "Connecting to"

    )

    print(url)

    response = requests.get(

        url,

        stream=True

    )

    client = sseclient.SSEClient(
        response
    )

    for event in client.events():

        try:

            data = json.loads(
                event.data
            )

            if (

                data["event_type"]

                ==

                "PROCESSING_COMPLETE"

            ):

                print()

                print(

                    "Processing Complete"

                )

                video_path_from_server = (

                    data["video_path"]

                )

                processing_finished = True

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

                f"Received "

                f"{data['frame_number_1']}"

            )

        except Exception as e:

            print(e)


def wait_until_finished():

    while not processing_finished:

        time.sleep(0.2)


def reconstruct_video():

    print()

    print("Reconstructing Video...")

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print("Cannot open video")

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_video = os.path.join(

        OUTPUT_DIR,

        f"{job_id}_output.mp4"

    )

    writer = cv2.VideoWriter(

        output_video,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )

    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number in received_frames:

            frame = received_frames.pop(
                frame_number
            )

            print(
                f"Replacing Frame "
                f"{frame_number}"
            )


        writer.write(frame)

        frame_number += 1

    cap.release()

    writer.release()

    print()

    print("Output Saved")

    print(output_video)


def main():

    upload_video()

    listener = threading.Thread(

        target=listen_sse,

        daemon=True

    )

    listener.start()

    wait_until_finished()

    reconstruct_video()


if __name__ == "__main__":

    main()