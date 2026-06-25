import cv2
import base64
import numpy as np
import requests
import sseclient
import json
import threading
import os

SSE_URL = "http://127.0.0.1:8000/stream-detections"

INPUT_VIDEO = "uploads/WIN_20260624_13_57_56_Pro.mp4"

OUTPUT_VIDEO = "outputs/reconstructed_output.mp4"

received_frames = {}

stream_finished = False
def decode_frame(encoded):

    image_bytes = base64.b64decode(encoded)

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    return frame

def listen_sse():

    global stream_finished

    print("Connecting to SSE...")

    response = requests.get(
        SSE_URL,
        stream=True
    )

    client = sseclient.SSEClient(response)

    for event in client.events():

        try:

            data = json.loads(
                event.data
            )

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
                f"Received frames "
                f"{data['frame_number_1']} "
                f"& "
                f"{data['frame_number_2']}"
            )

        except Exception as e:

            print(e)

    stream_finished = True


def reconstruct_video():

    cap = cv2.VideoCapture(
        INPUT_VIDEO
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
        "outputs",
        exist_ok=True
    )

    writer = cv2.VideoWriter(

        OUTPUT_VIDEO,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)

    )

    frame_number = 0

    print("Writing video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number in received_frames:

            frame = received_frames.pop(
                frame_number
            )

            print(
                f"Replacing frame "
                f"{frame_number}"
            )

        writer.write(frame)

        frame_number += 1

    cap.release()

    writer.release()

    print()

    print("Finished")

    print(
        OUTPUT_VIDEO
    )


if __name__ == "__main__":

    listener = threading.Thread(
        target=listen_sse,
        daemon=True
    )

    listener.start()

    reconstruct_video()