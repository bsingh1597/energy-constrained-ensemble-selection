import os
import shutil

# Paths
val_dir = '/home/myid/bs83243/mastersProject/ILSVRC/Data/CLS-LOC/val'
val_labels_file = '/home/myid/bs83243/mastersProject/ILSVRC/ImageSets/CLS-LOC/evaluation.txt'
output_dir = '/home/myid/bs83243/mastersProject/ILSVRC/Data/CLS-LOC/reorganized_val'

# Read val.txt (assuming it's in format 'image_name.jpg class_label')
with open(val_labels_file, 'r') as f:
    val_labels = f.readlines()

# Create class subdirectories and move images
for line in val_labels:
    image_name, _, class_label = line.strip().split()
    print(image_name, class_label)
    image_name += ".JPEG"

    # Create class subdirectory if it doesn't exist
    class_dir = os.path.join(output_dir, class_label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Move image to the corresponding class directory
    src = os.path.join(val_dir, image_name)
    dst = os.path.join(class_dir, image_name)
    shutil.move(src, dst)

print("Validation set reorganized into class-labeled subfolders.")
