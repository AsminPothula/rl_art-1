# needs to be reviewed - add proper comments 
import numpy as np
import cv2

def init_canvas(image_shape):
    return np.zeros(image_shape, dtype=np.uint8)

def update_canvas(canvas, start_point, end_point, color=[255, 255, 255], thickness=2, alpha=1.0, stroke_type='line', **kwargs):
    """
    Draws a shape on the given canvas based on the stroke_type with the specified color and thickness.

    Args:
        canvas (numpy.ndarray): The image or canvas on which the shape will be drawn.
        start_point (tuple): A tuple (x, y) representing the starting vector.
        end_point (tuple): A tuple (x, y) representing the ending vector.
        alpha (float, optional): The transparency of the shape. 0.0 is fully transparent, 1.0 is fully opaque. Defaults to 1.0.
        stroke_type (str, optional): The type of stroke to use. Can be 'line', 'rectangle', 'circle', 'arrowed_line', 'ellipse', or 'polygon'. Defaults to 'line'.
        color (tuple or int, optional): A tuple (B, G, R) representing the color of the shape in BGR format, or an int for grayscale. 
                                        Defaults to white (255, 255, 255).
        thickness (int, optional): The thickness of the shape in pixels. Defaults to 2.
        **kwargs: Additional arguments for customized stroke types.

    Returns:
        numpy.ndarray: The updated canvas with the shape drawn on it.
    """
    # Check if the canvas is grayscale (single channel)
    if len(canvas.shape) == 2 and len(color) == 3:
        # Convert color to grayscale
        color = int(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
    elif len(canvas.shape) == 3 and len(color) == 1:
        # Convert grayscale color to BGR
        color = (color[0], color[0], color[0])


    # For not we are only focusing on simple line strokes. In future, 
    # we can add more complex strokes like bezier curves, splines, etc. or 
    # even custom strokes., like painterly, brush strokes, etc.
    # For now, we will use OpenCV to draw the strokes.
    if stroke_type == 'line':
        cv2.line(canvas, start_point, end_point, color, thickness)
    # elif stroke_type == 'custom':
    #     custom_draw = kwargs.get('custom_draw')
    #     if callable(custom_draw):
    #         custom_draw(canvas, start_point, end_point, color, thickness, **kwargs)
    #     else:
    #         raise ValueError("For 'custom' stroke_type, 'custom_draw' must be a callable function.")
    else:
        raise ValueError(f"Unsupported stroke_type: {stroke_type}")

    return canvas

def save_canvas(canvas, path):
    cv2.imwrite(path, canvas)