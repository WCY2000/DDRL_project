import os
import random
import shutil

# Specify the source and destination folders
source_folder = './dataset/train'
destination_folder = './dataset/val'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all files in the source folder
all_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# Randomly select 30 images
selected_images = random.sample(all_files, 30)

# Copy the selected images to the destination folder
for image_name in selected_images:
    source_path = os.path.join(source_folder, image_name)
    destination_path = os.path.join(destination_folder, image_name)
    shutil.copy(source_path, destination_path)

print(f"Successfully copied {len(selected_images)} images to {destination_folder}")
