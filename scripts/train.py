from ultralytics import YOLO
import argparse
import os

parser = argparse.ArgumentParser(
                    prog='Yeah')

parser.add_argument('yaml', type=str, help="yaml file")
parser.add_argument('epochs', type=int, help="Number of epochs")
parser.add_argument('output', type=str, help="Name of the output directory")
parser.add_argument('model', type=str, help="Model to use e.g. yolov8n.pt")
parser.add_argument('device', type=int, help="Model to use e.g. yolov8n.pt")
parser.add_argument('lr', type=str, help="Model to use e.g. yolov8n.pt")
parser.add_argument('-p', action='store_true', help="Model to use e.g. yolov8n.pt")

args = parser.parse_args()

data = args.yaml
epochs = args.epochs
output = args.output
model_name = args.model
device_n = int(args.device)
pt = args.p
lr = float(args.lr)

print("Model used:", model_name)
model = YOLO(model_name)

results = model.train(
    data=data, 
    epochs=epochs,
    batch=16, 
    augment=False, 
    crop_fraction=0.0,
    name=output,
    pretrained=pt,
    lr0=lr,
    lrf=lr,
    patience=0,
    device=device_n,
)

metrics = model.val()

print(f"map50-95 {metrics.box.map}")
print(f"map50 {metrics.box.map50}")
print(f"map75 {metrics.box.map75}")
print(f"f1 {metrics.box.f1}")
print(f"p {metrics.box.p}")
print(f"r {metrics.box.r}")


os.makedirs("results", exist_ok=True)
with open(f"results/{output}.txt", "w") as f:
    f.write(f"{metrics.box.map},{metrics.box.map50},{metrics.box.map75},{metrics.box.f1},{metrics.box.p},{metrics.box.r}\n")