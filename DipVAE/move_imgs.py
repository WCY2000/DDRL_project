import os
import shutil

# The source directory where your images are currently stored
source_dir = "./dataset/val"

# The destination directory where you want to organize your images
destination_dir = "./dataset"

# Create a range for your demo numbers
demo_range = range(30, 67)  # This includes 0 to 50

for demo_num in demo_range:
    # Construct the directory names
    demo_dir = os.path.join(destination_dir, f"demo_{demo_num}")
    train_dir = os.path.join(demo_dir, "train")
    val_dir = os.path.join(demo_dir, "val")

    # Create the directories if they don't already exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Define the pattern for the filenames
    file_pattern = f"{demo_num}_*.jpg"

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.startswith(f"{demo_num}_") and filename.endswith(".jpg"):
            # Construct the full file paths
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(val_dir, filename)

            # Move the file to the appropriate 'val' directory
            shutil.move(source_path, destination_path)

print("Organization complete.")
