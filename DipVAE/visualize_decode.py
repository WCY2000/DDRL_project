import cv2
import os


def images_to_video(image_folder, output_video_file, fps=5, size=None):
    """
    Convert a series of images to a video.

    Parameters:
    - image_folder: Path to the folder containing the images.
    - output_video_file: Path to the output video file (should have .mp4 extension).
    - fps: Frame rate of the output video.
    - size: Size of each frame (width, height). If None, the size of the first image will be used.
    """
    # Get all image files in the folder, sorted numerically
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith(".png")],
        key=lambda x: int(x.split(".")[0]),
    )

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # For MP4 format
    if not size:
        # Determine the size of images by reading the first image
        first_image_path = os.path.join(image_folder, images[0])
        first_image = cv2.imread(first_image_path)
        size = (first_image.shape[1], first_image.shape[0])

    out = cv2.VideoWriter(output_video_file, fourcc, fps, size)

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        # Resize the image if it does not match the specified size
        if img.shape[1] != size[0] or img.shape[0] != size[1]:
            img = cv2.resize(img, size)

        # Write the frame to the video
        out.write(img)

    # Release the VideoWriter object
    out.release()


image_folder = "/home/chenyu/Desktop/visuotactile_representations/DipVAE/dataset/reconstructed_images_openteach_gt_check"  # Update this to your folder path
output_video_file = "./openteach.mp4"  # Update this to your desired output path
fps = 5  # Frame rate

# Create the video from images
images_to_video(image_folder, output_video_file, fps)
