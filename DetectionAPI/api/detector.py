from insightface.model_zoo import get_model

detector = get_model(
    "models/det_2.5g.onnx"
)

detector.prepare(
    ctx_id=-1,
    input_size=(640,640)
)

def detect_faces(frame):
    bboxes,_ = detector.detect(frame)
    return bboxes