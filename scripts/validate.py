from ultralytics import YOLO
import argparse
import os

parser = argparse.ArgumentParser(
                    prog='Yeah')

parser.add_argument('output', type=str, help="Name of the output directory")
parser.add_argument('model', type=str, help="Model to use e.g. yolov8n.pt")
parser.add_argument('-s', metavar='<some_value>', type=str, help="Where to save the results. Should point to a txt file")
parser.add_argument('-t', action='store_true', help="Where to save the results. Should point to a txt file")

args = parser.parse_args()

output = args.output
model_name = args.model

print("Model used:", model_name)
model = YOLO(model_name)

spl = "test" if args.t else "val"

print(f"Running validation on {spl} split...")
metrics = model.val(name=output, split=spl)

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

# Save to folder if the argument to do so is given
s = args.s
if s:
    with open(s, "w") as f:
        f.write("map5095 ,map50   ,map75   ,f1      ,f2      ,prec    ,recall  \n")
        f.write(f"{map5095:.6f},{map50:.6f},{map75:.6f},{f1:.6f},{f2:.6f},{p:.6f},{r:.6f}\n")
        f.write(f"{map5095:.6f}&{map50:.6f}&{map75:.6f}&{f1:.6f}&{f2:.6f}&{p:.6f}&{r:.6f}\n")