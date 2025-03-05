import enum
import os
from pathlib import Path
from threading import Thread
from time import time
import math
import cv2
from copy import deepcopy

from Mosse_Tracker.Mosse import MOSSE
from Mosse_Tracker.utils import draw_str
from Mosse_Tracker.utils import RectSelector

from System.Data.CONSTANTS import Work_Tracker_Interpolation

pi = 22/7

global id
id = 0
global frames
frames = []

class TrackerType(enum.Enum):
   MOSSE = 1
   DLIB = 2

class Tracker:
    def __init__(self, frame, cut_size, frame_width, frame_height, tracker_id=0, tracker_type=TrackerType.MOSSE):
        self.history = []
        self.tracker_type = tracker_type
        xmin, ymin, xmax, ymax = cut_size
        self.width, self.height = map(cv2.getOptimalDFTSize, [xmax - xmin, ymax - ymin])

        if tracker_type == TrackerType.MOSSE:
            self.tracker = MOSSE(frame, cut_size, learning_rate=0.225, psrGoodness=5)
            self.addHistory(self.tracker.getCutFramePosition())
        else:
            xmin, ymin, xmax, ymax = cut_size
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(frame, dlib.rectangle(int(xmin), int(ymin), int(xmax), int(ymax)))
            self.addHistory([xmin, ymin, xmax, ymax])
            self.dx = []
            self.dy = []

        xmin, ymin, xmax, ymax = cut_size
        self.vehicle_width, self.vehicle_height = map(cv2.getOptimalDFTSize, [xmax - xmin, ymax - ymin])
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracker_id = tracker_id
        self.index = 0
        self.avg_speed = [None]*30
        self.estimationFutureCenter = [-1]*30

    def addHistory(self, cut_size):
        """Add current cut frame in history for later use"""
        self.history.append(cut_size)

    def getHistory(self):
        """Get history in [[xmin,ymin,xmax,ymax]] format"""
        return self.history

    def update(self, frame):
        """Update the tracker to current frame and add the updated position to history"""
        if self.tracker_type == TrackerType.MOSSE:
            is_stopped = False
            if len(self.tracker.dx) >= 3 and Work_Tracker_Interpolation:
                if self.getAvgSpeed(len(self.tracker.dx)-3, len(self.tracker.dx)) < 20:
                    is_stopped = True

            self.tracker.updateTracking(frame, is_stopped)
            self.addHistory(self.tracker.getCutFramePosition())

        else:
            self.tracker.update(frame)
            if len(self.dx) == 0:
                self.dx.append(0)
                self.dy.append(0)
            else:
                x, y = self.get_position()
                xold, yold = self.get_position(self.history[-1])
                dx, dy = x - xold, y - yold
                self.dx.append(dx)
                self.dy.append(dy)
            self.addHistory(self.getCutFramePosition(self.get_position()))

        return self.history[-1]

    def getTrackerPosition(self):
        """Get last tracker position"""
        return self.history[-1]

    def getCutFramePosition(self, center):
        """Get the cut frame position (only for dlib tracker)"""
        if center == -1:
            center = self.center
        x = center[0]
        y = center[1]
        xmin = int(x - 0.5*(self.width-1))
        ymin = int(y - 0.5*(self.height-1))
        xmax = int(self.width+xmin)
        ymax = int(self.height+ymin)
        cut_size = [xmin, ymin, xmax, ymax]
        return cut_size

    def get_position(self, cut_size=None):
        """Get position from cut size (only for dlib tracker)"""
        if cut_size is None:
            pos = self.tracker.get_position()
            xmin = int(pos.left())
            ymin = int(pos.top())
            xmax = int(pos.right())
            ymax = int(pos.bottom())
        else:
            xmin, ymin, xmax, ymax = cut_size
        x = int(xmin + 0.5*self.width)
        y = int(ymin + 0.5*self.height)
        return (x, y)

    def getTrackedFramesBoxed(self, last_no_of_frame=0, after_no_of_frames=1):
        """Get dimensions of the history to make video clip later"""
        xmin = self.history[-after_no_of_frames][0]
        ymin = self.history[-after_no_of_frames][1]
        xmax = self.history[-after_no_of_frames][2]
        ymax = self.history[-after_no_of_frames][3]
        num_of_frames = len(self.history)
        if last_no_of_frame != 0:
            num_of_frames = last_no_of_frame

        size = len(self.history)
        for i in range(size-2, size-num_of_frames-1, -1):
            position = self.history[i]
            if position[0] < xmin:
                xmin = position[0]
            if position[1] < ymin:
                ymin = position[1]
            if position[2] > xmax:
                xmax = position[2]
            if position[3] > ymax:
                ymax = position[3]

        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, self.frame_width))
        ymax = int(min(ymax, self.frame_height))

        return xmin, ymin, xmax, ymax

    def showFrame(self, frame):
        """Visualize tracker on frame"""
        if self.tracker_type == TrackerType.MOSSE:
            (x, y) = self.tracker.getCenterOfTracker()
            xmin, ymin, xmax, ymax = self.tracker.getCutFramePosition()
        else:
            (x, y) = self.get_position()
            xmin, ymin, xmax, ymax = self.getCutFramePosition(self.get_position())

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255))

        if self.tracker_type == TrackerType.MOSSE:
            if self.tracker.isGood():
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            else:
                cv2.line(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255))
                cv2.line(frame, (xmax, ymin), (xmin, ymax), (0, 0, 255))

    def clearHistory(self):
        """Clear tracking history"""
        self.history = []

    def saveTracking(self, frames):
        """Save tracking results to video file"""
        new_frames, width, height, _, _, _, _ = self.getFramesOfTracking(frames)
        if new_frames is None:
            return
            
        os.makedirs('./track_videos', exist_ok=True)
        out = cv2.VideoWriter(f'./track_videos/{self.tracker_id}) {self.index}.avi', 
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

        for frame in new_frames:
            out.write(frame)
            
        self.index += 1
        out.release()

    def getMaxSpeed(self):
        """Get maximum speed of the tracked object"""
        if self.tracker_type == TrackerType.MOSSE:
            x = max(self.tracker.dx)
            y = max(self.tracker.dy)
        else:
            x = max(self.dx)
            y = max(self.dy)
            
        r = pow(pow(x, 2)+pow(y, 2), 0.5)
        r_coefficient = r * self.getCarSizeCoefficient()
        return r_coefficient
        
    def getAvgSpeed(self, from_frame_no=-1, to_frame_no=-1):
        """Get average speed of the tracked object"""
        if self.tracker_type == TrackerType.MOSSE:
            if from_frame_no == -1 or to_frame_no == -1:
                dx_change = self.tracker.dx
                dy_change = self.tracker.dy
            else:
                dx_change = self.tracker.dx[from_frame_no:to_frame_no]
                dy_change = self.tracker.dy[from_frame_no:to_frame_no]
        else:
            if from_frame_no == -1 or to_frame_no == -1:
                dx_change = self.dx
                dy_change = self.dy
            else:
                dx_change = self.dx[from_frame_no:to_frame_no]
                dy_change = self.dy[from_frame_no:to_frame_no]

        x = sum(dx_change)/len(dx_change)
        y = sum(dy_change)/len(dy_change)
        r = pow(pow(x, 2) + pow(y, 2), 0.5)
        r_coefficient = r * self.getCarSizeCoefficient()
        return r_coefficient

    def getCurrentSpeed(self):
        """Get current speed of the tracked object"""
        if self.tracker_type == TrackerType.MOSSE:
            no_of_last_frames = min(len(self.tracker.dx), 3)
            x = sum(self.tracker.dx[-no_of_last_frames:]) / no_of_last_frames
            y = sum(self.tracker.dy[-no_of_last_frames:]) / no_of_last_frames
        else:
            no_of_last_frames = min(len(self.dx), 3)
            x = sum(self.dx[-no_of_last_frames:]) / no_of_last_frames
            y = sum(self.dy[-no_of_last_frames:]) / no_of_last_frames
            
        r = pow(pow(x, 2) + pow(y, 2), 0.5)
        r_coefficient = r * self.getCarSizeCoefficient()
        return r_coefficient

    def getCarSizeCoefficient(self):
        """Calculate size coefficient for speed normalization"""
        if self.tracker_type == TrackerType.MOSSE:
            area = self.tracker.area
        else:
            area = self.width * self.height

        coefficient = 43200/area
        return coefficient

    def getCarAngle(self):
        """Calculate angle of movement in degrees"""
        if self.tracker_type == TrackerType.MOSSE:
            max_index_to_measure = min(1000, len(self.tracker.dx))
            dx = sum(self.tracker.dx[:max_index_to_measure])
            dy = sum(self.tracker.dy[:max_index_to_measure])
        else:
            max_index_to_measure = min(1000, len(self.dx))
            dx = sum(self.dx[:max_index_to_measure])
            dy = sum(self.dy[:max_index_to_measure])
            
        # Handle special cases
        if dx == 0:
            if dy > 0:
                return 270
            elif dy < 0:
                return 90
            else:
                return -1

        # Calculate angle
        degree = math.degrees(math.atan(abs(dy/dx)))
        
        # Adjust based on quadrant (remember y coordinates start at top)
        if dx < 0 and dy >= 0:
            return 180 + degree
        elif dx < 0 and dy <= 0:
            return 180 - degree
        elif dx > 0 and dy <= 0:
            return degree
        else:
            return 360 - degree

    def futureFramePosition(self):
        """Predict future position based on recent movement"""
        if self.tracker_type == TrackerType.MOSSE:
            if len(self.tracker.dx) < 5 or len(self.tracker.dx) > 20:
                self.estimationFutureCenter.append(self.tracker.center)
                return -1, -1, -1, -1
                
            measure = min(len(self.tracker.dx), 10)
            expectedPositionNo = len(self.tracker.dx) + 10
            x, y = self.tracker.center
            dx = sum(self.tracker.dx[-measure:]) / len(self.tracker.dx[-measure:])
            dy = sum(self.tracker.dy[-measure:]) / len(self.tracker.dy[-measure:])
            x_new = x + dx * measure
            y_new = y + dy * measure
            self.estimationFutureCenter[expectedPositionNo] = (x_new, y_new)
            return self.tracker.getCutFramePosition((x_new, y_new))
        else:
            if len(self.dx) < 5 or len(self.dx) > 20:
                self.estimationFutureCenter.append(self.get_position(self.history[-1]))
                return -1, -1, -1, -1
                
            measure = min(len(self.dx), 10)
            expectedPositionNo = len(self.dx) + 10
            x, y = self.get_position(self.history[-1])
            dx = sum(self.dx[-measure:]) / len(self.dx[-measure:])
            dy = sum(self.dy[-measure:]) / len(self.dy[-measure:])
            x_new = x + dx * measure
            y_new = y + dy * measure
            self.estimationFutureCenter[expectedPositionNo] = (x_new, y_new)
            return self.getCutFramePosition((x_new, y_new))

    def getFramesOfTracking(self, frames, last_no_of_frames=30):
        """Extract frames for crash detection analysis"""
        if len(self.history) < last_no_of_frames:
            return None, -1, -1, -1, -1, -1, -1
            
        xmin, ymin, xmax, ymax = self.getTrackedFramesBoxed(last_no_of_frames)
        width, height = xmax - xmin, ymax - ymin
        new_frames = []

        size = len(frames)
        for i in range(size - last_no_of_frames, size, 1):
            new_frames.append(frames[i][ymin:ymax, xmin:xmax])
            
        return new_frames, width, height, xmin, xmax, ymin, ymax

    def isAboveSpeedLimit(self, from_frame_no=-1, to_frame_no=-1):
        """Check if vehicle exceeds speed threshold"""
        if self.avg_speed[to_frame_no] is None:
            self.avg_speed[to_frame_no] = self.getAvgSpeed(from_frame_no, to_frame_no)
            
        return self.avg_speed[to_frame_no] > 50


class TrackerManager:
    def __init__(self, srcVid, paused=False, test=True):
        """Initialize tracker manager"""
        self.cap = cv2.VideoCapture(srcVid)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, self.frame = self.cap.read()
        
        if not ret:
            print(f"ERROR: Could not read from video source: {srcVid}")
            return
            
        cv2.imshow('frame', self.frame)
        self.rect_sel = RectSelector('frame', self.select)
        self.trackers = []
        self.paused = paused
        self.frames = []

    def select(self, rect):
        """Callback when user selects a rectangle to track"""
        global id
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = Tracker(frame_gray, rect, self.frame_width, self.frame_height, id, TrackerType.DLIB)
        id += 1
        self.trackers.append(tracker)

    def saveTrackers(self, trackers):
        """Save all trackers to video files"""
        for tracker in self.trackers:
            tracker.saveTracking(frames)

    def run(self):
        """Main tracking loop"""
        f = 1
        cum = 0
        global frames
        
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                    
                dim = (480, 360)
                self.frame = cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA)
                frames.append(self.frame.copy())
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                
                t = time()
                for tracker in self.trackers:
                    f += 1
                    tracker.update(frame_gray)
                    cum += time() - t

            # Visualization
            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.showFrame(vis)
            self.rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            ch = cv2.waitKey(10)
            
            if ch == 27:  # ESC
                break
            if ch == ord(' '):  # Space
                self.paused = not self.paused
            if ch == ord('c'):  # 'c'
                self.trackers = []
            if f % 30 == 0:
                thread = Thread(target=self.saveTrackers, args=(self.trackers,))
                thread.start()


if __name__ == '__main__':
    tracker_manager = TrackerManager(str(Path(__file__).parent.parent) + "\\videos\\1528.mp4", paused=True)
    tracker_manager.run()