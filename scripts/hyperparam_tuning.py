from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Tune hyperparameters on Pierre for 30 epochs
model.tune(data=".\Pierre.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)

