from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11n.pt")

# Tune hyperparameters on Pierre for 30 epochs
model.tune(data="./datasets/data/Pierre.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)

