from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    imgsz=640,
    epochs=80,
    batch=8,
    freeze=5
)
