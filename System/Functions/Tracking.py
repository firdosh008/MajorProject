import cv2

from Mosse_Tracker.TrackerManager import Tracker, TrackerType
from System.Data.CONSTANTS import Work_Tracker_Type_Mosse


class Tracking:
    def __init__(self):
        pass

    def track(self, frames, boxes, frame_width, frame_height):
        trackers = []
        trackerId = 0
        frame = frames[0]
        
        # Initialize a tracker for each detected box
        for _, box in enumerate(boxes):
            # Extract box coordinates
            xmin = int(box[1])
            xmax = int(box[2])
            ymin = int(box[3])
            ymax = int(box[4])

            # Convert to grayscale for tracking
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            trackerId += 1
            
            # Ensure coordinates are within frame boundaries
            xmax = min(xmax, frame_width - 1)
            ymax = min(ymax, frame_height - 1)

            # Create appropriate tracker type based on settings
            if Work_Tracker_Type_Mosse:
                trackers.append(Tracker(frame_gray, (xmin, ymin, xmax, ymax), 
                                        frame_width, frame_height, trackerId, TrackerType.MOSSE))
            else:
                trackers.append(Tracker(frame_gray, (xmin, ymin, xmax, ymax), 
                                        frame_width, frame_height, trackerId, TrackerType.DLIB))

        # Update trackers for each subsequent frame
        for i in range(1, len(frames)):
            frame = frames[i]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Update each tracker with the new frame
            for tracker in trackers:
                tracker.update(frame_gray)
                tracker.futureFramePosition()

        return trackers