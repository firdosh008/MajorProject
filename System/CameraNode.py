import copy
from time import time
import cv2
import threading
from System.Controller.JsonEncoder import JsonEncoder
from boxes.yoloFiles import loadFile


class CameraNode(threading.Thread):
    """
    Class that handles video input and feeds frames into the processing pipeline
    """
    
    def __init__(self, camera_id, file_path=None, files=True, city="None", district_no="None"):
        """
        Initialize camera node
        
        Args:
            camera_id: Unique identifier for this camera
            file_path: Path to video file to process
            files: Whether to use file-based detection boxes
            city: City location of the camera
            district_no: District number within the city
        """
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.read_file = files
        self.file_path = file_path
        self.no_of_frames = 0
        self.frame_width = 480
        self.frame_height = 360
        self.city = city
        self.district_no = district_no
        self.json_encoder = JsonEncoder()

    def run(self):
        """Main thread method that processes the video"""
        self.process_video_file()

    def process_video_file(self):
        """Process a video file and send frames to detection pipeline"""
        # Load pre-computed detection boxes if using file-based detection
        if self.read_file:
            fileBoxes = loadFile(self.file_path)

        # Open video file
        cap = cv2.VideoCapture(self.file_path)
        frames = []
        t = time()

        while True:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize to standard dimensions
            frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)

            self.no_of_frames += 1
            
            # When we have 30 frames, process them as a batch
            if len(frames) == 30:
                # Create a deep copy for processing
                new_frames_list = copy.deepcopy(frames)
                
                # Keep last 15 frames for next batch
                frames = frames[15:]
                
                # Get detection boxes for this batch
                new_boxes = []
                if self.read_file:
                    new_boxes = fileBoxes[self.no_of_frames - 30]

                # Send batch to processing pipeline
                self.json_encoder.feed(
                    self.camera_id,
                    self.no_of_frames - 29,  # Starting frame ID
                    new_frames_list,
                    self.frame_width,
                    self.frame_height,
                    self.read_file,
                    new_boxes,
                    self.city,
                    self.district_no
                )
                
            # Track frame rate
            if self.no_of_frames % 30 == 0:
                elapsed = time() - t
                t = time()

        # Release video resource
        cap.release()