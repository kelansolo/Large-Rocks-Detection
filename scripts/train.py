from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(
                    prog='Yeah')

parser.add_argument('yaml', type=str, help="yaml file")
parser.add_argument('epochs', type=int, help="Number of epochs")
parser.add_argument('output', type=str, help="Name of the output directory")
parser.add_argument('model', type=str, help="Model to use e.g. yolov8n.pt")

args = parser.parse_args()

data = args.yaml
epochs = args.epochs
output = args.output
model_name = args.model

print("Model used:", model_name)
model = YOLO(model_name)

results = model.train(
    data=data, 
    epochs=epochs,
    batch=16, 
    augment=False, 
    crop_fraction=0.0,
    name=output,
)

metrics = model.val()

print(f"map50-95 {metrics.box.map}")
print(f"map50 {metrics.box.map50}")
print(f"map75 {metrics.box.map75}")