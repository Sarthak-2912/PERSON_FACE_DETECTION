from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from sse_starlette.sse import (
    EventSourceResponse
)

import asyncio
import json
import os

from api.video_processor import (
    process_video,
    DETECTION_QUEUE
)

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


@app.post("/process-video")
async def process_video_api(
    video: UploadFile = File(...)
):

    video_path = os.path.join(
        UPLOAD_DIR,
        video.filename
    )

    with open(
        video_path,
        "wb"
    ) as f:

        f.write(
            await video.read()
        )

    asyncio.create_task(
        process_video(video_path)
    )

    return {
        "status": "processing_started",
        "video": video.filename
    }


@app.get("/stream-detections")
async def stream_detections():

    async def event_generator():

        while True:

            detection = (
                await DETECTION_QUEUE.get()
            )

            print(
                f"SSE SENT -> "
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