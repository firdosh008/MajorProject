import cv2
from PIL import Image

class Detection:
    def __init__(self, yolo):
        self.yolo = yolo

    def detect(self, frames, frame_width, frame_height, read_file, boxes_file=None, read_file_self=False, tf=True):
        boxes = []
        # detect vehicles
        if read_file_self:
            # From files
            boxes = boxes_file
        elif tf:
            img = Image.fromarray(frames[0])
            _, boxes = self.yolo.detect_image(img)
        else:
            # This branch is no longer used since we're removing PyTorch
            raise NotImplementedError("PyTorch detection has been removed")

        return boxes