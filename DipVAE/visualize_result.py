import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

image_pairs = range(0, 920)  # Assuming the images are numbered from 0 to 125
stitched_images = []

for i in image_pairs:
    img_path1 = f"/home/chenyu/Desktop/visuotactile_representations/DipVAE/dataset/reconstructed_LSTM_kinova_0404/{i:04d}.png"
    img_path2 = f"/home/chenyu/Desktop/visuotactile_representations/DipVAE/dataset/reconstructed_LSTM_Kinova_gt_0404/{i:04d}.png"

    # Read images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Optional: Resize images if they are not of the same dimensions

    # Annotate ground truth image
    position = (10, img2.shape[0] - 10)  # Adjust as needed
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1  # Adjust as needed
    color = (255, 255, 255)  # White
    thickness = 2
    img2 = cv2.putText(
        img2,
        "reconstructed from z",
        position,
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    # Stitch images
    stitched_img = np.hstack((img1, img2))
    stitched_images.append(stitched_img)

# Create a video from stitched images
clip = ImageSequenceClip(
    [img[:, :, ::-1] for img in stitched_images], fps=10
)  # Convert BGR to RGB
clip.write_videofile("./stitch.mp4", codec="libx264")
