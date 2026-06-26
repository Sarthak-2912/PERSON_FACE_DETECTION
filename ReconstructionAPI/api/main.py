from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Form
)

from fastapi.responses import FileResponse

import os
import cv2
import numpy as np

from api.reconstruction import (
    reconstruct_video,
    JOB_STATUS,
    REPLACEMENT_FRAMES
)

app = FastAPI(
    title="Video Reconstruction API"
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


@app.get("/health")
def health():

    return {
        "status": "running"
    }


# ----------------------------------------------------------
# Receive processed anomaly frames from Detection API
# ----------------------------------------------------------

@app.post("/upload-frame")
async def upload_frame(

    job_id: str = Form(...),

    frame_number: int = Form(...),

    frame: UploadFile = File(...)

):

    image_bytes = await frame.read()

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid frame"
        )

    if job_id not in REPLACEMENT_FRAMES:

        REPLACEMENT_FRAMES[job_id] = {}

    REPLACEMENT_FRAMES[job_id][frame_number] = image

    print(
        f"[{job_id}] Stored replacement frame {frame_number}"
    )

    return {

        "status": "frame_received",

        "job_id": job_id,

        "frame_number": frame_number

    }


# ----------------------------------------------------------
# Receive original video from Ingestion API
# ----------------------------------------------------------

@app.post("/upload-video")
async def upload_video(

    background_tasks: BackgroundTasks,

    job_id: str = Form(...),

    video: UploadFile = File(...)

):

    video_path = os.path.join(

        UPLOAD_DIR,

        f"{job_id}_{video.filename}"

    )

    with open(
        video_path,
        "wb"
    ) as f:

        f.write(
            await video.read()
        )

    JOB_STATUS[job_id] = "PROCESSING"

    background_tasks.add_task(

        reconstruct_video,

        job_id,

        video_path

    )

    return {

        "status": "processing_started",

        "job_id": job_id

    }


@app.get("/job-status/{job_id}")
def job_status(job_id: str):

    if job_id not in JOB_STATUS:

        raise HTTPException(

            status_code=404,

            detail="Invalid Job ID"

        )

    return {

        "job_id": job_id,

        "status": JOB_STATUS[job_id]

    }


@app.get("/download/{job_id}")
def download_video(job_id: str):

    output_video = os.path.join(

        OUTPUT_DIR,

        f"{job_id}_output.mp4"

    )

    if not os.path.exists(output_video):

        raise HTTPException(

            status_code=404,

            detail="Video not ready"

        )

    return FileResponse(

        output_video,

        media_type="video/mp4",

        filename=f"{job_id}_output.mp4"

    )