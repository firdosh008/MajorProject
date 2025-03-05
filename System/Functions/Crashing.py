import cv2

from Mosse_Tracker.TrackerManager import TrackerType
from System.Data.CONSTANTS import Work_Crash_Estimation_Only


class Crashing:
    """
    Class responsible for crash detection logic
    """
    
    def __init__(self, vif):
        self.vif = vif

    def crash(self, frames, trackers):
        """
        Main crash detection method that analyzes trackers for possible collisions
        
        Args:
            frames: List of video frames
            trackers: List of vehicle trackers
            
        Returns:
            crash_dimensions: Coordinates of crash area or empty list if no crash
        """
        crash_dimensions = []

        # Check all tracker pairs for potential collisions
        for i in range(len(trackers)):
            for j in range(i + 1, len(trackers)):
                if i == j:
                    continue
                    
                tracker_A = trackers[i]
                tracker_B = trackers[j]

                # Calculate distance threshold based on vehicle sizes
                asize = pow(pow(tracker_A.vehicle_height, 2) + pow(tracker_A.vehicle_width, 2), 0.5) * .25
                bsize = pow(pow(tracker_B.vehicle_height, 2) + pow(tracker_B.vehicle_width, 2), 0.5) * .25
                distance_threshold = asize + bsize

                # Check for collisions at different time points
                if (self.checkDistance(tracker_A, tracker_B, 16, distance_threshold) or
                    self.checkDistance(tracker_A, tracker_B, 19, distance_threshold) or
                    self.checkDistance(tracker_A, tracker_B, 22, distance_threshold) or
                    self.checkDistance(tracker_A, tracker_B, 25, distance_threshold) or
                    self.checkDistance(tracker_A, tracker_B, 28, distance_threshold)):
                
                    # Handle crash detection based on configuration
                    if Work_Crash_Estimation_Only:
                        self.crashEstimation(crash_dimensions, tracker_A, tracker_B, frames)
                    else:
                        crash_dimensions.extend(self.predict(frames, [tracker_B, tracker_A]))

        # Combine crash areas if multiple crashes detected
        if len(crash_dimensions) > 0:
            xmin = min(dim[0] for dim in crash_dimensions)
            ymin = min(dim[1] for dim in crash_dimensions)
            xmax = max(dim[2] for dim in crash_dimensions)
            ymax = max(dim[3] for dim in crash_dimensions)
            crash_dimensions = [xmin, ymin, xmax, ymax]

        return crash_dimensions

    def checkDistance(self, tracker_A, tracker_B, frame_no, distance_threshold):
        """
        Check if two trackers are in collision at a specific frame
        
        Args:
            tracker_A, tracker_B: The two trackers to check
            frame_no: The frame number to check
            distance_threshold: Minimum distance for collision
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Skip if neither vehicle is moving fast enough
        if not tracker_A.isAboveSpeedLimit(frame_no - 10, frame_no) and not tracker_B.isAboveSpeedLimit(frame_no - 10, frame_no):
            return False

        # Get predicted positions
        xa, ya = tracker_A.estimationFutureCenter[frame_no]
        xb, yb = tracker_B.estimationFutureCenter[frame_no]
        
        # Calculate distance between predicted positions
        r = pow(pow(xa - xb, 2) + pow(ya - yb, 2), 0.5)

        # Direct collision if distance is zero
        if r == 0:
            return True
        # Not a collision if distance is greater than threshold
        elif r > distance_threshold:
            return False

        # Get actual positions
        if tracker_A.tracker_type == TrackerType.MOSSE:
            xa_actual, ya_actual = tracker_A.tracker.centers[frame_no]
            xb_actual, yb_actual = tracker_B.tracker.centers[frame_no]
        else:
            xa_actual, ya_actual = tracker_A.get_position(tracker_A.history[frame_no])
            xb_actual, yb_actual = tracker_B.get_position(tracker_B.history[frame_no])
            
        # Calculate difference between actual and predicted positions
        difference_trackerA_actual_to_estimate = pow(pow(xa_actual - xa, 2) + pow(ya_actual - ya, 2), 0.5)
        difference_trackerB_actual_to_estimate = pow(pow(xb_actual - xb, 2) + pow(yb_actual - yb, 2), 0.5)
        max_difference = max(difference_trackerA_actual_to_estimate, difference_trackerB_actual_to_estimate)

        # If the difference is significant compared to the distance, consider it a collision
        return max_difference / r > 0.5

    def predict(self, frames_RGB, trackers):
        """
        Use VIF model to predict if a crash occurred
        
        Args:
            frames_RGB: List of color video frames
            trackers: List of vehicle trackers
            
        Returns:
            crash_dimensions: Coordinates of crash areas
        """
        gray_frames = self.convertToGrayFrames(frames_RGB)
        no_crash = 0
        crash = 0
        crash_dimensions = []
        
        for tracker in trackers:
            tracker_frames, width, height, xmin, xmax, ymin, ymax = tracker.getFramesOfTracking(gray_frames)
            crash_dimensions.append([xmin, ymin, xmax, ymax])

            # Skip if frames couldn't be extracted or frame is too small
            if tracker_frames is None:
                continue
            if xmax - xmin < 50:
                continue
            if ymax - ymin <= 28:
                continue
            if (ymax - ymin) / (xmax - xmin) < 0.35:
                continue

            # Run crash prediction model
            feature_vec = self.vif.process(tracker_frames)
            result = self.vif.clf.predict(feature_vec.reshape(1, 304))
            
            if result[0] == 0.0:
                no_crash += 1
            else:
                crash += 1
                tracker.saveTracking(frames_RGB)

        # Return empty list if no crash detected
        if crash == 0:
            crash_dimensions = []
            
        return crash_dimensions

    def convertToGrayFrames(self, frames_RGB):
        """Convert RGB frames to grayscale"""
        gray_frames = []
        for frame in frames_RGB:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return gray_frames

    def crashEstimation(self, crash_dimensions, tracker_A, tracker_B, frames):
        """
        Estimate crash dimensions based on trackers without using VIF model
        
        Args:
            crash_dimensions: List to store crash areas
            tracker_A, tracker_B: The two trackers involved in crash
            frames: List of video frames
        """
        # Process first tracker
        tracker_frames, width, height, xmin, xmax, ymin, ymax = tracker_A.getFramesOfTracking(
            self.convertToGrayFrames(frames))
            
        if not (xmax - xmin < 50 or ymax - ymin <= 28 or (ymax - ymin) / (xmax - xmin) < 0.35):
            crash_dimensions.extend([[xmin, ymin, xmax, ymax]])
            
        # Process second tracker
        tracker_frames, width, height, xmin, xmax, ymin, ymax = tracker_B.getFramesOfTracking(
            self.convertToGrayFrames(frames))
            
        if not (xmax - xmin < 50 or ymax - ymin <= 28 or (ymax - ymin) / (xmax - xmin) < 0.35):
            crash_dimensions.extend([[xmin, ymin, xmax, ymax]])