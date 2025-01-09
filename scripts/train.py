from ultralytics import YOLO
import argparse
import os

parser = argparse.ArgumentParser(
                    prog='Yeah')

parser.add_argument('yaml', type=str, help="yaml file")
parser.add_argument('epochs', type=int, help="Number of epochs")
parser.add_argument('output', type=str, help="Name of the output directory")
parser.add_argument('model', type=str, help="Model to use e.g. yolov8n.pt")
parser.add_argument('device', type=int, help="ID of the GPU to use")
parser.add_argument('lr', type=str, help="Learning rate (0.1 for default)")
parser.add_argument('-p', action='store_true', help="Add to set the pre-trained argument to true")

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

map5095 = metrics.box.map
map50 = metrics.box.map50
map75 = metrics.box.map75
f1 = metrics.box.f1[0]
p = metrics.box.p[0]
r = metrics.box.r[0]
f2 = (5 * p * r) / (4 * p + r + 1e-8)

print(f"map50-95 {map5095:.6f}")
print(f"map50 {map50:.6f}")
print(f"map75 {map75:.6f}")
print(f"f1 {f1:.6f}")
print(f"f2 {f2:.6f}")
print(f"p {p:.6f}")
print(f"r {r:.6f}")


os.makedirs("results", exist_ok=True)
with open(f"results/{output}.txt", "w") as f:
    f.write("map5095 ,map50   ,map75   ,f1      ,f2      ,prec    ,recall  \n")
    f.write(f"{map5095:.6f},{map50:.6f},{map75:.6f},{f1:.6f},{f2:.6f},{p:.6f},{r:.6f}\n")
    f.write(f"{map5095:.6f}&{map50:.6f}&{map75:.6f}&{f1:.6f}&{f2:.6f}&{p:.6f}&{r:.6f}")