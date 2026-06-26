HOST = "0.0.0.0"
PORT = 8000
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from sse_starlette.sse import (
    EventSourceResponse
)

import asyncio
import json
import os
import cv2
import numpy as np

from api.frame_processor import process_frame

from api.video_processor import JOB_QUEUES

app = FastAPI(
    title="Buffalo-M Detection API"
)

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@app.get("/health")
def health():

    return {
        "status": "running"
    }




@app.post("/process-frame-pair")
async def process_frame_pair_api(

    job_id: str = Form(...),

    frame_number_1: int = Form(...),

    frame_number_2: int = Form(...),
    
    timestamp_1: float = Form(...),
    
    timestamp_2: float = Form(...),

    frame1: UploadFile = File(...),

    frame2: UploadFile = File(...)

):

    if job_id not in JOB_QUEUES:

        JOB_QUEUES[job_id] = asyncio.Queue()

    # ---------- Decode frame 1 ----------

    image_bytes = await frame1.read()

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image1 = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    # ---------- Decode frame 2 ----------

    image_bytes = await frame2.read()

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image2 = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if image1 is None or image2 is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid frame."
        )

    await process_frame(
        image1,
        job_id,
        frame_number_1,
        timestamp_1
    )

    await process_frame(
        image2,
        job_id,
        frame_number_2,
        timestamp_2
    )

    return {

        "status": "processed",

        "frames": [

            frame_number_1,

            frame_number_2

        ]

    }

@app.post("/finish-detection")
async def finish_job(

    job_id: str = Form(...)

):

    if job_id not in JOB_QUEUES:

        raise HTTPException(

            status_code=404,

            detail="Invalid Job ID"

        )

    await JOB_QUEUES[job_id].put(

        {

            "event_type":

                "PROCESSING_COMPLETE",

            "job_id":

                job_id

        }

    )

    return {

        "status":

            "completed",

        "job_id":

            job_id

    }

@app.get("/stream-detections/{job_id}")
async def stream_detections(
    job_id: str
):

    if job_id not in JOB_QUEUES:

        raise HTTPException(
            status_code=404,
            detail="Invalid Job ID"
        )

    async def event_generator():

        queue = JOB_QUEUES[job_id]

        while True:

            detection = await queue.get()

            # Processing finished

            if detection.get(
                "event_type"
            ) == "PROCESSING_COMPLETE":

                yield {

                    "event": "completed",

                    "data": json.dumps(
                        detection
                    )

                }

                print(
                    f"Job {job_id} completed."
                )

                # cleanup

                del JOB_QUEUES[job_id]

                break

            print(

                f"[{job_id}] "

                f"SSE -> "

                f"{detection['timestamp']} sec"

            )

            yield {

                "event": "anomaly",

                "data": json.dumps(
                    detection
                )

            }

    return EventSourceResponse(
        event_generator()
    )