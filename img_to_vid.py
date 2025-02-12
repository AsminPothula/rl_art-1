import cv2
import os
import glob

# Define input folder (where images are stored)
image_folder = "pics"  # Change this if your folder has a different name
output_video = "output_video.mp4"  # Name of the output video

# Get list of image files (sorted in numerical order)
images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

# Read the first image to get the frame size
frame = cv2.imread(images[0])
height, width, layers = frame.shape
frame_size = (width, height)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
video = cv2.VideoWriter(output_video, fourcc, 10, frame_size)  # 10 FPS

# Add each image to the video
for img in images:
    frame = cv2.imread(img)
    video.write(frame)

# Release the video writer
video.release()

print(f"Video saved as {output_video}")
