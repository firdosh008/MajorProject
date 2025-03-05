import os
import cv2
from datetime import datetime
import json

from System.Controller.JsonEncoder import JsonEncoder
from System.Data.CONSTANTS import *
from System.Notifications.twilio_handler import TwilioHandler


class Master:
    def __init__(self):
        # Replace database with in-memory storage
        self.twilio_handler = TwilioHandler()
        self.crash_records = []
        self.saved_frames = {}  # camera_id -> {frame_id -> frames}
        
        # Create directory for saved videos if it doesn't exist
        if not os.path.exists('saved_crash_vid'):
            os.makedirs('saved_crash_vid')
        if not os.path.exists('saved_frames_vid'):
            os.makedirs('saved_frames_vid')

    def saveFrames(self, camera_id, starting_frame_id, frames, frame_width, frame_height):
        """Store frames in file system"""
        self.write(camera_id, frames, starting_frame_id, frame_width, frame_height, False)
        
        # Store frame info in memory
        if camera_id not in self.saved_frames:
            self.saved_frames[camera_id] = {}
        self.saved_frames[camera_id][starting_frame_id] = True

    def write(self, camera_id, frames, starting_frame_id, frame_width, frame_height, is_crash=False):
        """Write frames to video file"""
        folder = "saved_crash_vid" if is_crash else "saved_frames_vid"

        file_path = f'./{folder}/({camera_id}) {starting_frame_id}.avi'
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

        for frame in frames:
            out.write(frame)
            
        out.release()

    def getVideoFrames(self, camera_id, frame_id, is_crash=False):
        """Retrieve frames from video file"""
        folder = "saved_crash_vid" if is_crash else "saved_frames_vid"
        file_path = f'./{folder}/({camera_id}) {frame_id}.avi'
        
        cap = cv2.VideoCapture(file_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        return frames

    def recordCrash(self, camera_id, starting_frame_id, crash_dimensions):
        """Record crash event with visual marking"""
        new_frames = []
        from_no_of_times = PRE_FRAMES_NO

        # Collect frames from previous segments
        while from_no_of_times >= 0:
            last_frames = from_no_of_times * 30
            new_frames_id = starting_frame_id - last_frames
            if new_frames_id > 0:
                new_frames.extend(self.getVideoFrames(camera_id, new_frames_id, False))
                frame_width = len(new_frames[0][0])
                frame_height = len(new_frames[0])

            from_no_of_times -= 1

        # Unpack crash dimensions
        xmin, ymin, xmax, ymax = crash_dimensions

        # Determine how many frames to mark
        if len(new_frames) > 60:
            no_of_frames = 3
        elif len(new_frames) > 30:
            no_of_frames = 2
        else:
            no_of_frames = 1

        # Mark crash in earlier frames
        if len(new_frames) >= 60:
            for i in range(len(new_frames) - 60, len(new_frames) - 30, 6):
                cv2.rectangle(new_frames[i], (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)
                cv2.putText(new_frames[i], "Crash!", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
            no_of_frames = 3
            
        # Mark crash in recent frames
        for i in range(len(new_frames) - 30, len(new_frames), 1):
            fill = -1 if i % 2 == 0 else 2
            cv2.rectangle(new_frames[i], (xmin, ymin), (xmax, ymax), (0, 0, 255), fill)
            cv2.putText(new_frames[i], "Crash!", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)

        # Save crash video
        self.write(camera_id, new_frames, starting_frame_id, frame_width, frame_height, True)
        return no_of_frames

    def checkResult(self, camera_id, starting_frame_id, crash_dimentions, city, district_no):
        """Process crash detection results"""
        if len(crash_dimentions) == 0:
            return
            
        no_of_from_no = self.recordCrash(camera_id, starting_frame_id, crash_dimentions)
        
        # Store crash record in memory
        crash_time = datetime.utcnow()
        crash_record = {
            'camera_id': camera_id,
            'frame_id': starting_frame_id,
            'from_no': PRE_FRAMES_NO + 1,
            'city': city,
            'district': district_no,
            'crash_time': crash_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.crash_records.append(crash_record)
        
        # Save crash records for persistence
        self._save_crash_records()
        
        # Send notification
        self.sendNotification(camera_id, starting_frame_id, city, district_no)

    def _save_crash_records(self):
        """Save crash records to a JSON file"""
        try:
            with open('crash_records.json', 'w') as f:
                json.dump(self.crash_records, f, indent=2)
        except Exception as e:
            print(f"Error saving crash records: {e}")
            
    def _load_crash_records(self):
        """Load crash records from JSON file"""
        try:
            if os.path.exists('crash_records.json'):
                with open('crash_records.json', 'r') as f:
                    self.crash_records = json.load(f)
        except Exception as e:
            print(f"Error loading crash records: {e}")

    def sendNotification(self, camera_id, starting_frame_id, city, district_no):
        """Send notification about crash event"""
        jsonEncoder = JsonEncoder()
        date = f"{datetime.utcnow().date()} {str(datetime.utcnow().time()).split('.')[0]}"
        
        try:
            crash_pic = self.getCrashPhoto(camera_id, starting_frame_id)
        except Exception as e:
            print(f"Error getting crash photo: {str(e)}")
            crash_pic = None
        
        # Send Twilio notification
        self.twilio_handler.send_crash_alert(
            camera_id=camera_id,
            city=city,
            district_no=district_no,
            crash_pic=crash_pic
        )
        
        # Send notification to GUI
        jsonEncoder.sendNotification(camera_id, starting_frame_id, city, district_no, date, crash_pic)

    def executeQuery(self, start_date, end_date, start_time, end_time, city, district):
        """Search crash records based on query parameters"""
        # Load crash records if not already loaded
        if not self.crash_records:
            self._load_crash_records()
            
        # Parse date and time for comparison
        start_datetime_str = self._format_datetime(start_date, start_time)
        end_datetime_str = self._format_datetime(end_date, end_time)
        
        # Filter crash records
        filtered_records = []
        for record in self.crash_records:
            # Skip if city doesn't match
            if city and record['city'] != city:
                continue
                
            # Skip if district doesn't match
            if district and record['district'] != district:
                continue
                
            # Check if crash_time is in the requested range
            if start_datetime_str and end_datetime_str:
                if not (start_datetime_str <= record['crash_time'] <= end_datetime_str):
                    continue
                    
            filtered_records.append(record)
            
        # Sort by crash_time (newest first)
        filtered_records.sort(key=lambda x: x['crash_time'], reverse=True)
        
        self.replyQuery(filtered_records)
        
    def _format_datetime(self, date_str, time_str):
        """Format date and time strings to standard format for comparison"""
        try:
            if not date_str or not time_str:
                return None
                
            date_parts = date_str.split('/')
            if len(date_parts) != 3:
                return None
                
            day, month, year = date_parts
            
            # Pad with leading zeros
            if len(day) < 2:
                day = f"0{day}"
            if len(month) < 2:
                month = f"0{month}"
                
            time_parts = time_str.split(':')
            if len(time_parts) < 2:
                return None
                
            hour, minute = time_parts
            
            # Pad with leading zeros
            if len(hour) < 2:
                hour = f"0{hour}"
            if len(minute) < 2:
                minute = f"0{minute}"
                
            return f"{year}-{month}-{day} {hour}:{minute}:00"
        except Exception:
            return None

    def replyQuery(self, results):
        """Process crash records and send to GUI"""
        list_results = []
        
        for crash in results:
            camera_id = crash['camera_id']
            frame_id = crash['frame_id']
            city = crash['city']
            district = crash['district']
            crash_time = crash['crash_time']

            crash_pic = self.getCrashPhoto(camera_id, frame_id)
            sending_msg = {
                CAMERA_ID: camera_id,
                STARTING_FRAME_ID: frame_id,
                CITY: city,
                DISTRICT: district,
                CRASH_TIME: crash_time,
                CRASH_PIC: crash_pic
            }
            list_results.append(sending_msg)

        jsonEncoder = JsonEncoder()
        jsonEncoder.replyQuery(list_results)

    def getCrashPhoto(self, camera_id, starting_frame_id):
        """Extract a frame from crash video for thumbnail"""
        file_path = f'./saved_crash_vid/({camera_id}) {starting_frame_id}.avi'
        cap = cv2.VideoCapture(file_path)
        
        if cap is None or not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_no = min(89, total_frames - 1)
        if frame_no < 0:
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, photo = cap.read()
        cap.release()
        return photo if ret else None

    def sendVideoToGUI(self, camera_id, starting_frame_id):
        """Send video frames to GUI for playback"""
        video_frames = self.getVideoFrames(camera_id, starting_frame_id, True)
        jsonEncoder = JsonEncoder()
        jsonEncoder.replyVideo(video_frames)

    def sendRecentCrashesToGUI(self):
        """Send recent crashes to GUI"""
        # Load crash records if not already loaded
        if not self.crash_records:
            self._load_crash_records()
            
        # Sort by crash_time (newest first)
        recent_crashes = sorted(self.crash_records, key=lambda x: x['crash_time'], reverse=True)
        
        # Limit to 10 most recent
        recent_crashes = recent_crashes[:10]
        
        self.replyQuery(recent_crashes)