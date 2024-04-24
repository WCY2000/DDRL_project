import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import h5py

predicted_video = "/home/chenyu/Desktop/visuotactile_representations/DipVAE/stitch.mp4"
h5_file = "/home/chenyu/Desktop/visuotactile_representations/hiss/dataset/box_insertion_hiss_dataset_test/box_insertion_hiss_dataset_test_latent5t5_mu_image.h5"
frame_per_demo = 230
demo_num = 4


def add_black_images(image_list, target_length=230):
    image_list = list(image_list)

    num_images = len(image_list)
    print(num_images)
    if num_images >= target_length:
        return (
            image_list  # No need to add black images if the list is already long enough
        )

    black_image_shape = image_list[0].shape  # Assuming all images have the same shape
    black_image = np.zeros(black_image_shape, dtype=np.uint8)  # Create a black image

    # Calculate how many black images are needed to reach the target length
    num_black_images_needed = target_length - num_images

    # Append black images to the list
    for _ in range(num_black_images_needed):
        image_list.append(
            black_image.copy()
        )  # Make sure to copy black_image to prevent modifying it later

    return image_list


def extract_frames(video_path, target_size=(256, 256)):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames


def resize_images(images, target_height):
    resized_images = []
    for img in images:
        # Resize the image to have the target height while maintaining the aspect ratio
        scale_factor = target_height / img.shape[0]
        new_width = int(img.shape[1] * scale_factor)
        resized_img = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized_img)
    return resized_images


def stitch_images_side_by_side(l1, l2):
    # Assuming both lists have the same length
    num_images = len(l1)
    stitched_images = []

    for i in range(num_images):
        # Resize images to have the same height
        height = max(l1[i].shape[0], l2[i].shape[0])
        l1_resized = cv2.resize(
            l1[i], (int(l1[i].shape[1] * height / l1[i].shape[0]), height)
        )
        l2_resized = cv2.resize(
            l2[i], (int(l2[i].shape[1] * height / l2[i].shape[0]), height)
        )

        # Concatenate images horizontally
        stitched_image = np.concatenate((l1_resized, l2_resized), axis=1)
        stitched_images.append(stitched_image)

    return stitched_images


def create_video(image_list, output_video_path, fps=10):
    # Get the shape of the first image in the list
    height, width, _ = image_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for img in image_list:
        # Convert image depth to CV_8U
        img = cv2.convertScaleAbs(img)
        out.write(img)
    out.release()


predict_list = extract_frames(predicted_video)
total_image_list = []
with h5py.File(h5_file, "r") as file:
    for i in range(demo_num):
        image_list = file["image"][
            file["image_episode_ids"][i] : file["image_episode_ids"][i + 1]
        ]
        total_image_list += add_black_images(image_list)


create_video(
    stitch_images_side_by_side(predict_list, total_image_list), "./0404_LSTM_Kinova.mp4"
)
