import os
import json
import numpy as np
import cv2 as cv
from sksurgerynditracker.nditracker import NDITracker
import logging
from typing import Dict, Any
import time

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TrackerRecorder:
    def __init__(self, 
                 tracker_settings: Dict[str, Any],
                 file_path: str,
                 camera_id: int = 0,
                 num_markers: int = 2,
                 window_name: str = "Tracking Visualization"):
        """
        Initialize tracker and camera setup
        
        Args:
            tracker_settings: Dictionary containing NDI tracker settings
            file_path: Path to save recorded data
            camera_id: Camera device ID
            num_markers: Number of markers to track
            window_name: Name of the visualization window
        """
        logging.info("Initializing TrackerRecorder...")
        self.window_name = window_name
        self.file_path = file_path
        self.img_folder = os.path.join(file_path, 'img/')
        os.makedirs(self.img_folder, exist_ok=True)
        self.num_markers = num_markers
        
        self.data = {}
        self.latest_marker_data = [None] * num_markers  # Array for multiple markers
        
        # Initialize hardware
        self._init_tracker(tracker_settings)
        self._init_camera(camera_id)
        
        logging.info("TrackerRecorder initialized successfully.")

    def _init_tracker(self, settings: Dict[str, Any]) -> None:
        """Initialize NDI tracker with error handling"""
        try:
            logging.info(f"Initializing tracker with settings: {settings}")
            self.tracker = NDITracker(settings)
            self.tracker.start_tracking()
            logging.info("Tracker started successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize tracker: {str(e)}")
            raise RuntimeError(f"Failed to initialize tracker: {str(e)}")

    def _init_camera(self, camera_id: int) -> None:
        """Initialize camera with optimal settings"""
        logging.info(f"Initializing camera with ID: {camera_id}")
        
        # First try with 1280x720
        self.capture = cv.VideoCapture(camera_id, cv.CAP_DSHOW)  # Add CAP_DSHOW for Windows
        if not self.capture.isOpened():
            logging.error(f"Failed to open camera {camera_id}")
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set resolution before starting to stream
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Take a test frame
        ret, frame = self.capture.read()
        if ret:
            actual_height, actual_width = frame.shape[:2]
            logging.info(f"Got resolution: {actual_width}x{actual_height}")
            
            if (actual_width, actual_height) != (1280, 720):
                logging.warning("Failed to set 1280x720, falling back to 640x480")
                
                # Release and retry with 640x480
                self.capture.release()
                self.capture = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
                self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                
                ret, frame = self.capture.read()
                if ret:
                    actual_height, actual_width = frame.shape[:2]
                    logging.info(f"Fallback resolution: {actual_width}x{actual_height}")
        
        # Other settings
        self.capture.set(cv.CAP_PROP_FPS, 30)
        self.capture.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        
        # Verify stream is working
        for _ in range(5):  # Take a few test frames
            ret, frame = self.capture.read()
            if not ret:
                logging.warning("Failed to read frame during initialization")
            else:
                logging.info("Camera initialization successful")
                break

    def update_marker_data(self) -> bool:
        """
        Update the latest marker data for all markers
        Returns: True if all markers are detected
        """
        port_handles, timestamps, framenumbers, tracking, quality = self.tracker.get_frame()
        
        if not (tracking and timestamps and len(tracking) >= self.num_markers):
            self.latest_marker_data = [None] * self.num_markers
            return False
            
        all_detected = True
        for i in range(self.num_markers):
            if i < len(tracking) and not np.isnan(tracking[i]).any():
                self.latest_marker_data[i] = {
                    "timestamp": timestamps[0],
                    "quaternion": tracking[i][:4].tolist(),  # First 4 elements [w,x,y,z]
                    "translation": tracking[i][4:7].tolist() # Next 3 elements [x,y,z]
                }
            else:
                self.latest_marker_data[i] = None
                all_detected = False
                
        return all_detected

    def record(self, max_fps: int = 30) -> None:
        """Main visualization loop with frame rate control"""
        frame_time = 1000 // max_fps
        logging.info(f"Starting recording loop at {max_fps} FPS.")
        
        try:
            while True:
                # Update marker data
                all_markers_detected = self.update_marker_data()

                # Capture camera frame
                ret, frame = self.capture.read()
                img_ts = int(time.time() * 1000)

                if not ret:
                    logging.error("Failed to capture camera frame.")
                    break
                
                # Add marker detection status to frame
                for i in range(self.num_markers):
                    status_text = f"Marker {i+1}: {'DETECTED' if self.latest_marker_data[i] else 'NOT DETECTED'}"
                    cv.putText(frame, status_text, (10, 30*(i+1)), cv.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0) if self.latest_marker_data[i] else (0, 0, 255), 2)
                
                # Display frame
                cv.imshow(self.window_name, frame)

                # Handle keypress
                key = cv.waitKey(frame_time) & 0xFF
                if key == ord(' '):  # Space key pressed
                    if all_markers_detected:
                        # Save frame
                        img_name = os.path.join(self.img_folder, f'{img_ts}.png')
                        cv.imwrite(img_name, frame)

                        # Store data for all markers
                        self.data[str(img_ts)] = {
                            "markers": [
                                {
                                    "quaternion": marker_data["quaternion"],  # Already a list
                                    "translation": marker_data["translation"]  # Already a list
                                } for marker_data in self.latest_marker_data if marker_data is not None
                            ],
                            "img_path": img_name
                        }
                        
                        # Log the saved data for verification
                        logging.info(f"Saved frame {img_ts} with marker data:")
                        for i, marker in enumerate(self.latest_marker_data):
                            if marker:
                                logging.info(f"Marker {i}:")
                                logging.info(f"  Quaternion: {marker['quaternion']}")
                                logging.info(f"  Translation: {marker['translation']}")
                    else:
                        logging.warning("Cannot save frame: Not all markers detected!")
                        print("Cannot save frame: Not all markers detected!")

                elif key == 27:  # ESC key pressed
                    break

        finally:
            self.cleanup()

    def save_data(self) -> None:
        """Save tracking data to a JSON file"""
        json_path = os.path.join(self.file_path, 'tracking_data.json')
        logging.info(f"Saving tracking data to {json_path}...")
        try:
            with open(json_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            logging.info("Tracking data saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save tracking data: {str(e)}")

    def cleanup(self) -> None:
        """Clean up resources and save data"""
        logging.info("Cleaning up resources...")
        self.tracker.stop_tracking()
        self.tracker.close()
        self.capture.release()
        cv.destroyAllWindows()
        
        # Save collected data
        self.save_data()
        logging.info("Resources cleaned up successfully.")

def main():
    """Main entry point with error handling"""
    settings_vega = {
        "tracker type": "polaris",
        "ip address": "192.168.2.17",
        "port": 8765,
        "romfiles": [
            "C:\\Users\\Callahan\\Downloads\\StarTrack_4.rom",
            # "C:\\Users\\Callahan\\Downloads\\8700339.rom"  # Add second marker ROM file
        ],
        "use quaternions": True
    }

    file_path = "./recorded_data_1112_2/"

    try:
        recorder = TrackerRecorder(settings_vega, file_path, num_markers=1)
        recorder.record()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
