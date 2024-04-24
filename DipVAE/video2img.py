import cv2
import os

for demo_num in range(100, 560):
    # Path to your AVI file
    avi_file_path = f"/nas/datasets/relay_kitchen_dataset/observations_seq_img_multiview/{demo_num:03d}_view0.mp4"

    # Directory where you want to save the images
    output_dir = "./dataset/train"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(avi_file_path)

    frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Save the frame as an image
        output_image_path = os.path.join(
            output_dir, f"{demo_num}_{frame_count:04d}.jpg"
        )
        cv2.imwrite(output_image_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_count} images.")
