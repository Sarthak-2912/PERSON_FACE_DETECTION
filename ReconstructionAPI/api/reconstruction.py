import cv2
import os
import time

OUTPUT_DIR = "outputs"

JOB_STATUS = {}

# Stores processed frames received from Detection API
# Structure:
# {
#     job_id: {
#         frame_number: frame
#     }
# }
REPLACEMENT_FRAMES = {}


def reconstruct_video(
    job_id,
    video_path
):

    received_frames = REPLACEMENT_FRAMES.get(
        job_id,
        {}
    )

    print()
    print(
        f"[{job_id}] Reconstruction Started"
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"[{job_id}] Cannot open video"
        )

        JOB_STATUS[job_id] = "FAILED"

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

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

    replaced_frames = 0

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number in received_frames:

            replacement = received_frames[frame_number]

            frame = replacement["frame"]

            timestamp = replacement["timestamp"]

            replaced_frames += 1

            print(
                f"[{job_id}] "
                f"Replacing Frame {frame_number} "
                f"at {timestamp:.3f}s"
            )

        writer.write(
            frame
        )

        frame_number += 1

    cap.release()

    writer.release()

    elapsed = round(
        time.time() - start_time,
        2
    )

    JOB_STATUS[job_id] = "COMPLETED"

    # Cleanup stored frames
    REPLACEMENT_FRAMES.pop(
        job_id,
        None
    )

    print()
    print(
        f"[{job_id}] Reconstruction Finished"
    )

    print(
        f"[{job_id}] Output : {output_video}"
    )

    print(
        f"[{job_id}] Frames Replaced : {replaced_frames}"
    )

    print(
        f"[{job_id}] Reconstruction Time : {elapsed} sec"
    )

    return output_video