# needs to be reviewed - add proper comments 
import cv2

def render_stroke(canvas, start, direction, radius):
    """
    Draw a line from start point to new point on circle defined by direction * radius.
    """
    end = (
        int(start[0] + direction[0] * radius),
        int(start[1] + direction[1] * radius)
    )
    cv2.line(canvas, start, end, color=255, thickness=1)
    return end  # return new point (becomes next start)
