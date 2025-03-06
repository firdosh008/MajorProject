from twilio.rest import Client
import os
from datetime import datetime
import cv2
import numpy as np
import re

class TwilioHandler:
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_PHONE_NUMBER')
        self.recipient_number = os.getenv('RECIPIENT_PHONE_NUMBER')
        
        # Format phone numbers properly for Twilio
        self.from_number = self._format_phone_number(self.from_number)
        self.recipient_number = self._format_phone_number(self.recipient_number)
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            print("Warning: Twilio credentials not found in environment variables")
            self.client = None

    def _format_phone_number(self, phone_number):
        """Format phone number to E.164 format (+1XXXXXXXXXX)"""
        if not phone_number:
            return None
            
        # Remove any non-digit characters except for leading '+'
        if phone_number.startswith('+'):
            digits_only = '+' + re.sub(r'\D', '', phone_number[1:])
        else:
            digits_only = '+' + re.sub(r'\D', '', phone_number)
            
        # Make sure US numbers have proper format
        if len(re.sub(r'\D', '', digits_only)) == 10 and not digits_only.startswith('+1'):
            digits_only = '+1' + re.sub(r'\D', '', digits_only)
            
        return digits_only

    def _save_temp_image(self, image_data):
        """Save numpy array image to temporary file"""
        if isinstance(image_data, np.ndarray):
            temp_path = "temp_crash.jpg"
            cv2.imwrite(temp_path, image_data)
            return temp_path
        return None

    def send_crash_alert(self, camera_id, city, district_no, crash_pic=None):
        if not self.client:
            print("Twilio client not initialized. Skipping notifications.")
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_body = (
            f"🚨 CRASH ALERT!\n"
            f"Time: {timestamp}\n"
            f"Location: {city}, {district_no}\n"
            f"Camera ID: {camera_id}"
        )

        # Handle crash_pic if it's a numpy array
        media_url = None
        if crash_pic is not None:
            temp_image_path = self._save_temp_image(crash_pic)
            if temp_image_path:
                print(f"Image saved to {temp_image_path}")
                # media_url = "https://your-cloud-storage/temp_crash.jpg"

        # Send SMS
        if self.from_number and self.recipient_number:
            try:
                message = self.client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=self.recipient_number
                )
                print(f"Crash alert SMS sent successfully! SID: {message.sid}")
                return True
            except Exception as e:
                print(f"Failed to send crash alert SMS: {str(e)}")
                return False
        
        return False