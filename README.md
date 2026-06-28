# Real-Time Multi-Person Face Detection and Video Reconstruction System

A distributed microservice-based computer vision system that performs **real-time face detection** using the **InsightFace Buffalo-M** model and reconstructs the original video by replacing anomaly frames with processed frames.

The system is designed using **three independent FastAPI microservices**, enabling modular deployment, scalability, and real-time communication between services.

---

# Features

* 🎥 Live webcam recording
* ⚡ Real-time frame streaming
* 👥 Multi-person face detection using Buffalo-M
* 🖼 Automatic bounding box generation
* 🔄 Real-time anomaly frame transmission
* 🎬 Original video preservation
* 🎞 Intelligent video reconstruction
* ⏱ Frame timestamp generation
* 🔢 Frame number tracking
* 📡 Server-Sent Events (SSE) for live anomaly notifications
* 🆔 Unique Job ID for every recording session
* 🧵 Concurrent processing support for multiple jobs
* 🏗 Microservice-based architecture

---

# System Architecture

The project is divided into three independent APIs.

## 1. Ingestion API

Responsibilities:

* Open webcam
* Record original video
* Capture live frames
* Sample the latest two frames every 0.5 seconds
* Generate frame numbers
* Generate timestamps
* Send frame pairs to Detection API
* Upload original video to Reconstruction API

---

## 2. Detection API

Responsibilities:

* Receive frame pairs
* Decode incoming frames
* Run Buffalo-M face detection
* Detect multiple persons
* Draw bounding boxes
* Generate anomaly events
* Send processed frames to Reconstruction API
* Publish Server-Sent Events (SSE)

---

## 3. Reconstruction API

Responsibilities:

* Receive original video
* Receive processed anomaly frames
* Store replacement frames
* Replace matching frames in original video
* Generate reconstructed output video

---

# Processing Workflow

1. User starts recording.
2. Ingestion API opens the webcam.
3. Original video recording begins.
4. Every 0.5 seconds the latest two frames are selected.
5. Frame numbers and timestamps are generated.
6. Frame pairs are sent to Detection API.
7. Buffalo-M detects faces.
8. If multiple faces are detected:

   * Bounding boxes are drawn.
   * Processed frame is sent to Reconstruction API.
   * SSE event is generated.
9. User stops recording.
10. Original video is uploaded to Reconstruction API.
11. Reconstruction API replaces anomaly frames.
12. Final reconstructed video is generated.

---

# Project Structure

```text
person_face_detection/

│
├── IngestionAPI/
│   ├── api/
│   │   ├── camera.py
│   │   ├── sender.py
│   │   └── ...
│   └── main.py
│
├── DetectionAPI/
│   ├── api/
│   │   ├── detector.py
│   │   ├── frame_processor.py
│   │   ├── video_processor.py
│   │   └── ...
│   └── main.py
│
├── ReconstructionAPI/
│   ├── api/
│   │   ├── reconstruction.py
│   │   └── ...
│   └── main.py
│
├── models/
│   └── det_2.5g.onnx
│
├── uploads/
├── outputs/
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python
* FastAPI
* OpenCV
* InsightFace Buffalo-M
* ONNX Runtime (CPU)
* NumPy
* Requests
* Server-Sent Events (SSE)
* UUID
* ThreadPoolExecutor

---

# Prerequisites

* Python 3.10+
* Webcam
* pip
* Virtual Environment (recommended)
* Buffalo-M ONNX model
* Git

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Navigate to the project

```bash
cd person_face_detection
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Model Setup

Download the Buffalo-M detection model and place it inside:

```text
models/
    det_2.5g.onnx
```

---

# Running the Project

Start the APIs in the following order.

### Detection API

```bash
cd DetectionAPI

python -m uvicorn api.main:app --reload --port 8000
```

---

### Reconstruction API

```bash
cd ReconstructionAPI

python -m uvicorn api.main:app --reload --port 8001
```

---

### Ingestion API

```bash
cd IngestionAPI

python -m uvicorn api.main:app --reload --port 8002
```

---

# API Endpoints

## Ingestion API

| Endpoint              | Description            |
| --------------------- | ---------------------- |
| POST /start-recording | Start webcam recording |
| POST /stop-recording  | Stop recording         |

---

## Detection API

| Endpoint                        | Description                  |
| ------------------------------- | ---------------------------- |
| POST /process-frame-pair        | Receive frame pairs          |
| POST /finish-detection          | Notify processing completion |
| GET /stream-detections/{job_id} | SSE anomaly stream           |
| GET /health                     | Health check                 |

---

## Reconstruction API

| Endpoint                 | Description                     |
| ------------------------ | ------------------------------- |
| POST /upload-frame       | Receive processed anomaly frame |
| POST /upload-video       | Receive original recorded video |
| GET /job-status/{job_id} | Check processing status         |
| GET /download/{job_id}   | Download reconstructed video    |
| GET /health              | Health check                    |

---

# Data Flow

```
Webcam
      │
      ▼
Ingestion API
      │
      ├── Original Video ─────────────► Reconstruction API
      │
      └── Frame Pair + Timestamp ─────► Detection API
                                           │
                                           ▼
                                 Buffalo-M Detection
                                           │
                                           ▼
                             Processed Anomaly Frame
                                           │
                                           ▼
                               Reconstruction API
                                           │
                                           ▼
                             Final Output Video
```

---

# Output

The project generates:

Original recorded video

```text
uploads/<job_id>.mp4
```

Reconstructed video

```text
outputs/<job_id>_output.mp4
```

The `uploads/` and `outputs/` directories are created automatically at runtime if they do not already exist.

---

# Future Improvements

* Docker support
* Docker Compose deployment
* GPU inference
* Redis message queue
* RabbitMQ / Kafka integration
* Kubernetes deployment
* Authentication
* Database persistence
* Monitoring and logging dashboard
* Cloud deployment

---

# License

This project is intended for educational and research purposes.

---

# Author

**Sarthak Bhardwaj**

B.Tech Computer Science & Engineering (Cloud Computing)

Built as a real-time distributed computer vision system using FastAPI, OpenCV, and InsightFace Buffalo-M.
