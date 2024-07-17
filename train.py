
# train dota v1
from ultralytics import YOLO

# Create a new YOLOv8n-OBB model from scratch
model = YOLO("yolov8m-obb.yaml")

# Train the model on the DOTAv2 dataset
results = model.train(data="/usr/src/TrafficNight/TrafficNight.yaml", 
                      batch=10, epochs=200, imgsz=640, cache=True,
                      auto_augment=True)