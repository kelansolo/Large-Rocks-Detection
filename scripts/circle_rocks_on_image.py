import matplotlib.patches as patches
import matplotlib.pyplot as plt

img_name = '2781_1141_3_0'

f, ax = plt.subplots(1,1)

img = plt.imread(f'data/{img_name}.tif')
ax.imshow(img)


with open(f'data/{img_name}.txt', 'r') as f:
    bboxes = f.readlines()
    for box in bboxes:
        x,y = box.split(' ')[1:3] 
        x = int(float(x) * 640)
        y = int(float(y) * 640)
        # Draw the square as a patch (red square) of size 20 pixel (10m)
        rect = patches.Circle((x, y), radius=20,  linewidth=2, edgecolor='r', facecolor='none')         
        ax.add_patch(rect)