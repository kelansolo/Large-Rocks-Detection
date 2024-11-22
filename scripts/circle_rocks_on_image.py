import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

dir = 'data'
img_dir = os.path.join(dir, 'images')



images = os.listdir(img_dir)

for image in images:
    f, ax = plt.subplots(1,1)
    img_name = image.split(".")[0]
    img_path = os.path.join(img_dir, f'{img_name}.tif')
    lbl_path = os.path.join(dir, "labels", f'{img_name}.txt')

    img = plt.imread(img_path)
    ax.imshow(img)



    with open(lbl_path, 'r') as f:
        bboxes = f.readlines()
        for box in bboxes:
            x,y = box.split(' ')[1:3] 
            x = int(float(x) * 640)
            y = int(float(y) * 640)
            # Draw the square as a patch (red square) of size 20 pixel (10m)
            rect = patches.Circle((x, y), radius=20,  linewidth=2, edgecolor='r', facecolor='none')         
            ax.add_patch(rect)

        plt.show()