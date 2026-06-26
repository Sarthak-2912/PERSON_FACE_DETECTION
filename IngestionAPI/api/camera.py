import cv2
import os
import uuid
import threading
import time

from api.sender import (
    dispatch_reconstruction,
    send_frame_pair,
    finish_detection
)

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

SHOW_PREVIEW = True


class CameraManager:

    def __init__(self):

        self.cap = None

        self.writer = None

        self.recording = False

        self.thread = None

        self.video_path = None

        self.job_id = None

        self.start_time = None

        self.fps = 30

        self.width = 640

        self.height = 480

        self.last_dispatch_time = 0
        
        self.buffer_frames = []

    def start_recording(self):

        if self.recording:

            return False

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            raise Exception(
                "Unable to open webcam"
            )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.fps <= 0:

            self.fps = 30

        self.width = int(

            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )

        )

        self.height = int(

            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )

        )

        self.job_id = str(
            uuid.uuid4()
        )

        self.video_path = os.path.join(

            UPLOAD_DIR,

            f"{self.job_id}.mp4"

        )

        self.writer = cv2.VideoWriter(

            self.video_path,

            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),

            self.fps,

            (
                self.width,
                self.height
            )

        )

        self.recording = True

        self.start_time = time.time()

        self.thread = threading.Thread(

            target=self.record,

            daemon=True

        )

        self.thread.start()

        print()

        print("=================================")
        print("Camera Recording Started")
        print("=================================")

        return True

    def record(self):

        frame_number = 0

        self.last_dispatch_time = time.time()

        self.buffer_frames = []

        while self.recording:

            ret, frame = self.cap.read()

            if not ret:

                print("Camera disconnected.")
                self.recording = False
                break


            self.writer.write(frame)


            self.buffer_frames.append((frame.copy(), frame_number))
            if len(self.buffer_frames) > 10:
                self.buffer_frames.pop(0)


            current_time = time.time()

            if current_time - self.last_dispatch_time >= 0.5:

                if len(self.buffer_frames) >= 2:

                    frame1, number1 = self.buffer_frames[-2]
                    frame2, number2 = self.buffer_frames[-1]
                    timestamp1 = round(number1 / self.fps, 3)
                    timestamp2 = round(number2 / self.fps, 3)

                    threading.Thread(
                        target=send_frame_pair,
                        args=(
                            frame1,
                            frame2,
                            self.job_id,
                            number1,
                            number2,
                            timestamp1,
                            timestamp2
                        ),
                        daemon=True
                    ).start()

                    self.buffer_frames.clear()

                self.last_dispatch_time = current_time

            frame_number += 1

            if SHOW_PREVIEW:

                cv2.imshow(
                    "Live Camera Feed",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):

                    self.recording = False
                    break

        print("Recording thread stopped.")
    
    def stop_recording(self):

        if not self.recording:

            return None

        self.recording = False

        if (

            self.thread

            and

            threading.current_thread() != self.thread

        ):

            self.thread.join()

        if self.cap:

            self.cap.release()

            self.cap = None

        if self.writer:

            self.writer.release()

            self.writer = None

        if SHOW_PREVIEW:

            cv2.destroyAllWindows()

        duration = round(

            time.time()

            - self.start_time,

            2

        )

        print()

        print("=================================")
        print("Recording Finished")
        print("=================================")

        print(f"Video : {self.video_path}")
        print(f"Job ID : {self.job_id}")
        print(f"Duration : {duration} sec")

        reconstruction_thread = threading.Thread(

            target=dispatch_reconstruction,

            args=(

                self.video_path,

                self.job_id

            ),

            daemon=True

        )

        reconstruction_thread.start()
        reconstruction_thread.join()

        finish_detection(
            self.job_id
    )

        return {

            "job_id": self.job_id,

            "video_path": self.video_path,

            "duration": duration

        }


camera = CameraManager()