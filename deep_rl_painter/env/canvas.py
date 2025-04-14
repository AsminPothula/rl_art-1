# needs to be reviewed - add proper comments 
import numpy as np
import cv2

def init_canvas(image_shape):
    return np.zeros(image_shape, dtype=np.uint8)

def update_canvas(canvas, start_point, end_point):
    cv2.line(canvas, start_point, end_point, color=255, thickness=1)
    return canvas

def save_canvas(canvas, path):
    cv2.imwrite(path, canvas)
