import os
import shutil
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk, ImageDraw

# Define the folder paths
input_folder = r"C:\Users\Kelan\Documents\1_EPFL\IPEO\LargeRocksDetectionDataset\LargeRocksDetectionDataset\swissImage_50cm_patches"
second_band_folder = r"C:\Users\Kelan\Documents\1_EPFL\IPEO\LargeRocksDetectionDataset\LargeRocksDetectionDataset\swissSURFACE3D_hillshade_patches"  # Folder with the second band (different band)


good_folder = r"C:\Users\Kelan\Documents\1_EPFL\IPEO\Good"
bad_folder = r"C:\Users\Kelan\Documents\1_EPFL\IPEO\Bad"

# Create good and bad folders if they don't exist
os.makedirs(good_folder, exist_ok=True)
os.makedirs(bad_folder, exist_ok=True)

# Load all TIFF images from the input folder
images = [file for file in os.listdir(input_folder) if file.lower().endswith('.tif')]
image_index = 0

def parse_annotations(annotation_path):
    """Parse annotation file and return a list of bounding boxes."""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # Ensure valid format
            class_id, x, y, w, h = map(float, parts)
            boxes.append((x, y, w, h))
    return boxes

def draw_annotations(image, annotations):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    for x, y, _, _ in annotations:
        # Convert normalized coordinates to pixel values
        x = int(float(x) * 640)
        y = int(float(y) * 640)

        # Draw a rectangle around the bounding box
        draw.circle([x,y], radius=20, outline="red", width=3)
    return image

def save_and_next(is_good):
    """Save the current image and annotation to the selected folder."""
    global image_index
    if image_index < len(images):
        current_image = images[image_index]
        annotation_file = current_image.replace('.tif', '.txt')
        source_image = os.path.join(input_folder, current_image)
        source_annotation = os.path.join(input_folder, annotation_file)
        
        target_folder = good_folder if is_good else bad_folder

        if os.path.exists(source_image):
            shutil.move(source_image, os.path.join(target_folder, current_image))
        else:
            print(f"Image file not found: {source_image}")
        
        if os.path.exists(source_annotation):
            shutil.move(source_annotation, os.path.join(target_folder, annotation_file))
        else:
            print(f"Annotation file not found: {source_annotation}")
        
        image_index += 1
        show_image()

def show_image():
    """Display the current image with annotations from both bands."""
    global image_index
    if image_index < len(images):
        current_image = images[image_index]
        annotation_file = current_image.replace('.tif', '.txt')
        
        # Paths for both bands
        img_path1 = os.path.join(input_folder, current_image)  # Image from the first band
        img_path2 = os.path.join(second_band_folder, current_image)  # Image from the second band
        
        # Open both images
        img1 = Image.open(img_path1).convert("RGB")  # First band
        img2 = Image.open(img_path2).convert("RGB")  # Second band
        
        # Parse and draw annotations on the first image
        annotation_path = os.path.join(input_folder, annotation_file)
        if os.path.exists(annotation_path):
            annotations = parse_annotations(annotation_path)
            img1 = draw_annotations(img1, annotations)
            img2 = draw_annotations(img2, annotations)  # Apply same annotations on the second band
        else:
            print(f"No annotations found for {current_image}")
        
        # Resize images to fit side by side
        img1.thumbnail((800, 600))
        img2.thumbnail((800, 600))
        
        # Combine the images horizontally
        combined_img = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))
        
        # Convert to Tkinter-compatible format
        img_tk = ImageTk.PhotoImage(combined_img)
        
        # Display image and update status
        label.config(image=img_tk)
        label.image = img_tk
        status_label.config(text=f"Image {image_index + 1} of {len(images)}: {current_image}")
    else:
        status_label.config(text="No more images!")
        label.config(image='')
        label.image = None
        good_button.config(state="disabled")
        bad_button.config(state="disabled")

# Create the GUI
root = Tk()
root.title("Image Sorter with Annotations")

# Display area for the image
label = Label(root)
label.pack()

# Status label
status_label = Label(root, text="")
status_label.pack()

# Buttons for classification
good_button = Button(root, text="Good", command=lambda: save_and_next(True))
good_button.pack(side="left", padx=10, pady=10)

bad_button = Button(root, text="Bad", command=lambda: save_and_next(False))
bad_button.pack(side="left", padx=40, pady=10)

# Start the app
show_image()
root.mainloop()