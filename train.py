from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.yaml')
 
# Training.
results = model.train(
   data='Pierre.yaml',
   epochs=1,
   batch=16,
   name='yolov8n_v8_50e',
   #pretrained = True,
   #augment = False,
   #crop_fraction = None,
   #workers = 0

)