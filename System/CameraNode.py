import copy
from time import time

import cv2
import threading
from System.Controller.JsonEncoder import JsonEncoder
from boxes.yoloFiles import loadFile


class CameraNode(threading.Thread):

    def __init__(self, camera_id, file_path=None, files=True, city="None", district_no="None"):
        threading.Thread.__init__(self)
        self.camera_id = camera_id  # special id for every camera
        self.read_file = files  # do you want to read detected cars from file?
        self.file_path = file_path  # file path if you want to detect cars from file
        self.no_of_frames = 0  # current no of frames processed by the camera
        self.frame_width = 480  # camera resolution from frame width
        self.frame_height = 360  # camera resolution from frame height
        self.city = city
        self.district_no = district_no
        self.json_encoder = JsonEncoder()

    def run(self):
        self.process_video_file()  # Changed from startStreaming to be more explicit

    def process_video_file(self):
        """Process a video file rather than handling streaming"""
        if self.read_file:
            fileBoxes = loadFile(self.file_path)  # return boxes of cars in the frames

        cap = cv2.VideoCapture(self.file_path)
        frames = []
        boxes = []
        t = time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)

            self.no_of_frames += 1
            if len(frames) == 30:
                new_frames_list = []
                new_frames_list = copy.deepcopy(frames)
                frames = frames[15:]  # Keep last 15 frames for next batch
                
                new_boxes = []
                if self.read_file:
                    new_boxes = fileBoxes[self.no_of_frames - 30]

                # Send to processing pipeline
                self.json_encoder.feed(self.camera_id, self.no_of_frames - 29, new_frames_list, 
                                       self.frame_width, self.frame_height, self.read_file, 
                                       new_boxes, self.city, self.district_no)
                
                print(int(self.no_of_frames / 30))
                
            if self.no_of_frames % 30 == 0:
                current_time = time() - t
                print(max(1 - current_time, 0))
                t = time()

        # Don't forget to release the capture when done
        cap.release()