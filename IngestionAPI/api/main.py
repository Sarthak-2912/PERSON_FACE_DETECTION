from fastapi import (
    FastAPI,
    HTTPException
)

from api.camera import (
    camera
)

app = FastAPI(
    title="Video Ingestion API"
)


@app.get("/health")
def health():

    return {

        "status": "running"

    }


@app.post("/start-camera")
def start_camera():

    try:

        if camera.recording:

            return {

                "status": "already_recording"

            }

        camera.start_recording()

        return {

            "status": "camera_started"

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


@app.post("/stop-camera")
def stop_camera():

    if not camera.recording:

        raise HTTPException(

            status_code=400,

            detail="Camera is not recording."

        )

    result = camera.stop_recording()

    return {

        "status": "processing_started",

        "job_id": result["job_id"],

        "video_path": result["video_path"],

        "duration": result["duration"]

    }


@app.get("/camera-status")
def camera_status():

    return {

        "recording": camera.recording,

        "job_id": camera.job_id

    }