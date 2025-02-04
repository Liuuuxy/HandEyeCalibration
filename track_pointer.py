import os
import json
import numpy as np
import cv2 as cv
from sksurgerynditracker.nditracker import NDITracker
import logging
from typing import Dict, Any
import time
from scipy.spatial.transform import Rotation as R

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PointerVisulizer:
    def __init__(
        self,
        tracker_settings: Dict[str, Any],
        file_path: str,
        camera_id: int = 0,
        num_markers: int = 2,
        window_name: str = "Tracking Visualization",
        fps: int = 10,
    ):
        """Initialize recorder with video settings"""
        self.fps = fps
        self.recording = False
        self.frame_count = 0

        # Initialize as before
        logging.info("Initializing PointerVisulizer...")
        self.window_name = window_name
        self.file_path = file_path
        self.video_folder = os.path.join(file_path, "video/")
        os.makedirs(self.file_path, exist_ok=True)
        os.makedirs(self.video_folder, exist_ok=True)
        self.num_markers = num_markers

        self.data = {}
        self.latest_marker_data = [None] * num_markers

        self.pointer_translation = self.pointer_translation = np.array(
            [[-18.24668526], [0.27911252], [-158.19044532]]  # Single list argument
        ).reshape(3, 1)

        # Initialize hardware
        self._init_tracker(tracker_settings)
        self._init_camera(camera_id)

        self.paths = []

        logging.info("PointerVisulizer initialized successfully.")

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
        self.capture = cv.VideoCapture(
            camera_id, cv.CAP_DSHOW
        )  # Add CAP_DSHOW for Windows
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

        self.camera_intrinsics = np.array(
            [
                [1.31108397e03, 0.00000000e00, 6.86486022e02],
                [0.00000000e00, 1.31801722e03, 4.41608492e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        # np.load("camera_matrix_0129_1_PARK.npy")
        self.camera_distortion = np.array(
            [-0.21975247, -0.01772734, 0.00826037, -0.00210365, -0.24672817]
        )

        # np.load("dist_coeffs_0129_1_PARK.npy")
        # self.T_marker_to_camera = np.load("T_cam2marker_0129_1_PARK.npy")
        self.T_marker_to_camera = np.array(
            [
                [4.45038364e-01, 1.01815503e-01, -8.89704703e-01, -2.65384336e02],
                [8.94482122e-01, -2.91752819e-03, 4.47094198e-01, 1.48419097e02],
                [4.29253819e-02, -9.94799021e-01, -9.23705588e-02, -1.88705508e01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

    def update_marker_data(self) -> bool:
        """
        Update the latest marker data for all markers
        Returns: True if all markers are detected
        """
        port_handles, timestamps, framenumbers, tracking, quality = (
            self.tracker.get_frame()
        )

        if not (tracking and timestamps and len(tracking) >= self.num_markers):
            self.latest_marker_data = [None] * self.num_markers
            return False
        # print(f"len(tracking): {len(tracking)}")

        all_detected = True
        for i in range(self.num_markers):
            if i < len(tracking) and not np.isnan(tracking[i]).any():
                tracking_data = tracking[i].flatten()
                self.latest_marker_data[i] = {
                    "timestamp": timestamps[0],
                    "quaternion": tracking_data[:4].tolist(),
                    "translation": tracking_data[4:7].tolist(),
                }
            else:
                self.latest_marker_data[i] = None
                all_detected = False
        # print(f"all_detected: {all_detected}")

        try:
            marker2 = self.latest_marker_data[1]
            if marker2 is None:
                # logging.warning(
                #     "Marker 2 not detected, skipping transformation matrix calculation."
                # )
                return False

            # print(f"Marker 2 data: {marker2}")
            if len(marker2["quaternion"]) != 4 or len(marker2["translation"]) != 3:
                logging.error(
                    f"Invalid quaternion or translation for marker 2: {marker2}"
                )
                return False

            # Get transformation matrix from marker 2 data
            T_pointer = self.get_transformation_matrix(
                marker2["translation"], marker2["quaternion"]
            )

            # Compute the pointer tip position as a homogeneous 4x1 vector.
            # NOTE: np.vstack([self.pointer_translation, 1]) converts the 3x1 offset to a 4x1 homogeneous vector.
            # breakpoint()
            pointer_tip_hom = T_pointer @ np.vstack([self.pointer_translation, 1])
            # Convert to a 3-element vector (drop the homogeneous coordinate)
            pointer_tip = pointer_tip_hom[:3, 0]
            # pointer_tip = T_pointer[:3, 3]
            # Append the 3D point to self.paths.
            self.paths.append(pointer_tip)
            if len(self.paths) > 100:
                self.paths = self.paths[-100:]
        except Exception as e:
            logging.error(f"Error processing marker 2 transformation: {str(e)}")
            raise

        return all_detected

    def get_transformation_matrix(self, position, quaternion):
        """
        Create a 4x4 transformation matrix from position and quaternion.
        """
        try:
            if len(position) != 3 or len(quaternion) != 4:
                raise ValueError("Invalid position/quaternion dimensions")
            # Rearrange the quaternion to match the scipy Rotation convention: [x, y, z, w]
            rotation_matrix = R.from_quat(
                [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
            ).as_matrix()
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = position
            return transformation_matrix
        except Exception as e:
            logging.error(f"Error in get_transformation_matrix: {str(e)}")
            raise

    def show_path(self, frame):
        try:
            overlay = np.zeros_like(frame)

            if self.latest_marker_data[0] is None:
                logging.warning("Marker 1 not detected, skipping path drawing.")
                return
            # breakpoint()
            T_polaris_to_marker = self.get_transformation_matrix(
                self.latest_marker_data[0]["translation"],
                self.latest_marker_data[0]["quaternion"],
            )
            T_polaris_to_camera = T_polaris_to_marker @ self.T_marker_to_camera
            T_camera_to_polaris = np.linalg.inv(T_polaris_to_camera)

            marker_paths = []

            # Draw fading path
            for T_polaris_to_pointer in self.paths:
                rotation_matrix_polaris_to_camera = T_camera_to_polaris[:3, :3]
                position_polaris_to_camera = T_camera_to_polaris[:3, 3]

                rvec, _ = cv.Rodrigues(rotation_matrix_polaris_to_camera)
                tvec = position_polaris_to_camera.reshape(3, 1)

                point_3d = np.array(T_polaris_to_pointer, dtype=np.float32).reshape(
                    1, 3
                )

                try:
                    uv_distorted, _ = cv.projectPoints(
                        point_3d,
                        rvec,
                        tvec,
                        self.camera_intrinsics,
                        self.camera_distortion,
                    )

                    u, v = uv_distorted.ravel()
                except Exception as e:
                    logging.warning(f"Projection failed: {str(e)}")
                    continue

                u, v = int(round(u)), int(round(v))
                # print(f"Marker position: ({u}, {v})")

                marker_paths.append((u, v))
            print(f"pointer position: {marker_paths[-1]}")

            PATH_COLOR = (0, 255, 0)

            MAX_OPACITY = 0.7

            for j in range(1, len(marker_paths)):
                alpha = (j / len(marker_paths)) * MAX_OPACITY

                temp_overlay = np.zeros_like(frame)

                cv.line(
                    temp_overlay,
                    marker_paths[j - 1],
                    marker_paths[j],
                    PATH_COLOR,
                    thickness=3,
                )

                overlay = cv.addWeighted(temp_overlay, alpha, overlay, 1.0, 0)

            frame = cv.addWeighted(overlay, 1.0, frame, 1.0, 0)

            if marker_paths:
                cv.circle(frame, marker_paths[-1], 5, (0, 255, 0), -1)

            return frame
        except Exception as e:
            logging.error(f"Error in show_path: {str(e)}")
            raise

    def record(self) -> None:
        """Main recording loop"""
        frame_time = 1000 // self.fps  # Time per frame in milliseconds

        try:
            while True:
                # Update marker data
                all_markers_detected = self.update_marker_data()
                ret, frame = self.capture.read()
                frame_ts = int(time.time() * 1000)  # Current timestamp

                if not ret:
                    logging.error("Failed to capture camera frame.")
                    break

                # Add recording status and marker detection status to frame
                status_color = (0, 255, 0) if self.recording else (0, 0, 255)
                cv.putText(
                    frame,
                    f"Recording: {self.recording}",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2,
                )

                for i in range(self.num_markers):
                    status_text = f"Marker {i+1}: {'DETECTED' if self.latest_marker_data[i] else 'NOT DETECTED'}"
                    cv.putText(
                        frame,
                        status_text,
                        (10, 60 + 30 * i),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0) if self.latest_marker_data[i] else (0, 0, 255),
                        2,
                    )

                if all_markers_detected:
                    frame = self.show_path(frame)

                # Display frame
                cv.imshow(self.window_name, frame)

                # Handle keypresses
                key = cv.waitKey(frame_time) & 0xFF

                if key == ord("r"):  # Start/Stop recording
                    if not self.recording:
                        # Start new recording
                        video_timestamp = int(time.time())
                        video_filename = os.path.join(
                            self.video_folder, f"video_{video_timestamp}.mp4"
                        )
                        self.video_writer = cv.VideoWriter(
                            video_filename,
                            cv.VideoWriter_fourcc(*"mp4v"),
                            self.fps,
                            (frame.shape[1], frame.shape[0]),
                        )
                        self.recording = True
                        self.frame_count = 0
                        self.data[video_timestamp] = {
                            "video_path": video_filename,
                            "frames": {},
                        }
                        logging.info(f"Started recording to {video_filename}")
                    else:
                        # Stop recording
                        self.video_writer.release()
                        self.recording = False
                        logging.info("Stopped recording")

                elif key == 27:  # ESC key
                    break

                # Save frame and marker data if recording
                if self.recording:
                    self.video_writer.write(frame)
                    self.frame_count += 1

                    # Save marker data for this frame
                    self.data[video_timestamp]["frames"][frame_ts] = {
                        "frame_number": self.frame_count,
                        "markers": [
                            (
                                {
                                    "quaternion": marker_data["quaternion"],
                                    "translation": marker_data["translation"],
                                }
                                if marker_data
                                else None
                            )
                            for marker_data in self.latest_marker_data
                        ],
                    }

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources and save data"""
        logging.info("Cleaning up resources...")
        self.tracker.stop_tracking()
        self.tracker.close()
        self.capture.release()
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()
        cv.destroyAllWindows()

        # Save collected data
        self.save_data()
        logging.info("Resources cleaned up successfully.")

    def save_data(self) -> None:
        """Save tracking data to a JSON file"""
        json_path = os.path.join(self.file_path, "tracking_data.json")
        logging.info(f"Saving tracking data to {json_path}...")
        try:
            with open(json_path, "w") as f:
                json.dump(self.data, f, indent=4)
            logging.info("Tracking data saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save tracking data: {str(e)}")


def main():
    """Main entry point with error handling"""
    settings_polaris = {
        "tracker type": "polaris",
        "romfiles": [
            "C:\\Users\\Callahan\\Downloads\\StarTrack_4.rom",
            # "C:\\Users\\Callahan\\Downloads\\8700339.rom",
            "C:\\Users\\Callahan\\Downloads\\8700340-Polaris_Passive_4-Marker_Probe.rom",  # Add second marker ROM file
        ],
        "use quaternions": True,
    }

    file_path = "handEyeCalib_data/recorded_data_0203_1/"

    try:
        recorder = PointerVisulizer(settings_polaris, file_path, num_markers=2)
        recorder.record()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
