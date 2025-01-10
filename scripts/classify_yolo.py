from ultralytics import YOLO

model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="./dataset_cls_good", epochs=400, imgsz=640, name="cls_200_good", patience=0, augment=False)

print("############################### Validating ################################")
metrics = model.val()

print(metrics)

print()
print("################################ Testing ##################################")

test = model.val(split='test')

print(test)