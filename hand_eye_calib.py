import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R


class CalibrationDataCollector:
    def __init__(self, checkerboard_size=(6, 6), square_size=10):
        """
        Initialize the calibration data collector

        Args:
            checkerboard_size: Tuple of (rows, cols) internal corners of the checkerboard
            square_size: Size of each square in millimeters (e.g., 5 for 5mm)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.calibration_data = {
            "marker_poses": [],
            "camera_poses": [],
            "camera_matrix": None,
            "dist_coeffs": None,
        }

        # Prepare object points for checkerboard in millimeters
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : checkerboard_size[0], 0 : checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size

    def load_data_from_visualizer(self, data_path):
        """
        Load images and corresponding marker poses saved by TrackerVisualizer

        Args:
            data_path: Path to the directory where TrackerVisualizer saved images and poses.json
        """
        # Load poses.json file
        poses_file = Path(data_path) / "tracking_data.json"
        with open(poses_file, "r") as f:
            pose_data = json.load(f)

        # Load each image and corresponding pose
        for ts, entry in pose_data.items():
            # print(entry)
            img_path = entry["img_path"]

            marker_data = entry["markers"][0]["quaternion"][0]

            # marker_data = entry['quaternion'][0]  # Get the first (and only) element
            marker_quaternion = marker_data[0:4]  # First 4 elements are quaternion
            marker_position = marker_data[4:7]

            # Read the image
            image = cv2.imread(str(img_path))
            print(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            # Add image and pose to calibration data
            success = self.add_calibration_frame(
                image, marker_position, marker_quaternion
            )
            if success:
                print(f"Successfully processed {img_path}")
            else:
                print(f"Failed to find checkerboard in {img_path}")

    def add_calibration_frame(self, image, marker_position, marker_quaternion):
        """
        Add a new frame to the calibration data

        Args:
            image: BGR image from camera
            marker_position: [x, y, z] position of marker in tracker frame
            marker_quaternion: [w, x, y, z] quaternion of marker in tracker frame

        Returns:
            bool: True if frame was successfully added
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

        if ret:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Store marker pose
            self.calibration_data["marker_poses"].append(
                {"position": marker_position, "quaternion": marker_quaternion}
            )

            # Store image points for camera calibration
            if "image_points" not in self.calibration_data:
                self.calibration_data["image_points"] = []
            self.calibration_data["image_points"].append(corners2)

            # Store object points
            if "object_points" not in self.calibration_data:
                self.calibration_data["object_points"] = []
            self.calibration_data["object_points"].append(self.objp)

            return True
        return False

    def calibrate_camera(self, image_size):
        """
        Perform camera calibration using collected frames

        Args:
            image_size: Tuple of (width, height) of the images

        Returns:
            bool: True if calibration was successful
        """
        if len(self.calibration_data["image_points"]) < 3:
            print("Need at least 3 valid frames for calibration")
            return False

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.calibration_data["object_points"],
            self.calibration_data["image_points"],
            image_size,
            None,
            None,
        )

        if ret:
            print("Camera calibration successful")
            print(f"Number of poses: {len(rvecs)}")  # Debug print
            print("\nCamera Calibration Matrix:")
            print(mtx)  # Add this line to print the camera matrix
            print("\nDistorsion Matrix:")
            print(dist)
            self.calibration_data["camera_matrix"] = mtx.tolist()
            self.calibration_data["dist_coeffs"] = dist.tolist()
            self.calibration_data["camera_poses"] = []  # Reset camera poses

            # Convert rotation vectors and translation vectors to transformation matrices
            for rvec, tvec in zip(rvecs, tvecs):
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()
                self.calibration_data["camera_poses"].append(T.tolist())

            # breakpoint()
            print(
                f"Number of marker poses: {len(self.calibration_data['marker_poses'])}"
            )
            print(
                f"Number of camera poses: {len(self.calibration_data['camera_poses'])}"
            )
            return True
        return False

    def solve_hand_eye_calibration(self):
        """
        Solve the hand-eye calibration using collected data

        Returns:
            numpy.ndarray: 4x4 transformation matrix from marker to camera
        """
        if len(self.calibration_data["camera_poses"]) < 2:
            raise ValueError("Need at least 2 poses for hand-eye calibration")

        print(
            f"Starting hand-eye calibration with {len(self.calibration_data['camera_poses'])} poses"
        )

        # Prepare matrices for hand-eye calibration
        R_gripper2base = []  # Rotation matrices from marker motions
        t_gripper2base = []  # Translation vectors from marker motions
        R_target2cam = []  # Rotation matrices from camera motions
        t_target2cam = []  # Translation vectors from camera motions

        # Calculate relative transformations
        for i in range(len(self.calibration_data["marker_poses"])):
            print(f"Processing pose pair {i} and {i+1}")

            marker1 = self.calibration_data["marker_poses"][i]

            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_quat(
                [
                    marker1["quaternion"][1],
                    marker1["quaternion"][2],
                    marker1["quaternion"][3],
                    marker1["quaternion"][0],
                ]
            ).as_matrix()
            T1[:3, 3] = marker1["position"]

            T_camera = np.array(self.calibration_data["camera_poses"][i])

            R_gripper2base.append(T1[:3, :3])
            t_gripper2base.append(T1[:3, 3])
            R_target2cam.append(T_camera[:3, :3])
            t_target2cam.append(T_camera[:3, 3])

        print(f"Number of relative transformations: {len(R_gripper2base)}")
        print(
            f"Shapes: R_gripper2base={len(R_gripper2base)}, t_gripper2base={len(t_gripper2base)}"
        )
        print(
            f"Shapes: R_target2cam={len(R_target2cam)}, t_target2cam={len(t_target2cam)}"
        )

        # Solve hand-eye calibration
        R_cam2marker, t_cam2marker = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK,
        )

        # Create transformation matrix
        T_cam2marker = np.eye(4)
        T_cam2marker[:3, :3] = R_cam2marker
        T_cam2marker[:3, 3] = t_cam2marker.flatten()

        self.T_cam2marker = T_cam2marker

        result_mag = np.sqrt(sum(x * x for x in t_cam2marker.flatten()))
        print(f"\nFinal translation magnitude: {result_mag:.2f} mm")

        return T_cam2marker

    def save_calibration_data(self, filepath):
        """Save calibration data to JSON file"""
        json_data = {}
        for key, value in self.calibration_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, list):
                # Handle lists that might contain numpy arrays
                json_data[key] = [
                    (
                        [
                            item.tolist() if isinstance(item, np.ndarray) else item
                            for item in sublist
                        ]
                        if isinstance(sublist, list)
                        else (
                            sublist.tolist()
                            if isinstance(sublist, np.ndarray)
                            else sublist
                        )
                    )
                    for sublist in value
                ]
            else:
                json_data[key] = value

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=4)

    def load_calibration_data(self, filepath):
        """Load calibration data from JSON file"""
        with open(filepath, "r") as f:
            self.calibration_data = json.load(f)

    def calculate_reprojection_errors(self, base_filename):
        """
        Calculate reprojection errors for both camera calibration and hand-eye calibration.

        Returns:
            dict: Dictionary containing mean errors and detailed error statistics
        """
        if (
            not self.calibration_data["camera_matrix"]
            or not self.calibration_data["dist_coeffs"]
        ):
            raise ValueError("Camera must be calibrated first")

        camera_matrix = np.array(self.calibration_data["camera_matrix"])
        dist_coeffs = np.array(self.calibration_data["dist_coeffs"])

        # camera_matrix = np.array([[1305.5, 0, 599.1], [0, 1325.2, 508.5], [0, 0, 1]])
        # dist_coeffs = np.array([-0.2720, 0.0549, 0.0128, 0.0145, 0.1170])

        # 1. Camera Calibration Reprojection Error
        total_error = 0
        per_view_errors = []

        for i in range(len(self.calibration_data["object_points"])):
            obj_points = self.calibration_data["object_points"][i]
            img_points = self.calibration_data["image_points"][i]

            # Project object points using camera calibration
            rvec = cv2.Rodrigues(
                np.array(self.calibration_data["camera_poses"][i])[:3, :3]
            )[0]
            tvec = np.array(self.calibration_data["camera_poses"][i])[:3, 3]

            projected_points, _ = cv2.projectPoints(
                obj_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_points = projected_points.astype(img_points.dtype)

            # print("img_points:", img_points.shape, img_points.dtype)
            # print("projected_points:", projected_points.shape, projected_points.dtype)

            # Calculate error for this view
            error = cv2.norm(img_points, projected_points, cv2.NORM_L2) / len(
                projected_points
            )
            per_view_errors.append(error)
            total_error += error

        mean_camera_error = total_error / len(self.calibration_data["object_points"])

        # 2. Hand-Eye Calibration Reprojection Error (if available)
        hand_eye_errors = []
        projected_points_all = []  # Store all projected points
        ground_truth_points_all = []  # Store all ground truth points

        if hasattr(self, "T_cam2marker"):
            rotation_matrices = []
            translation_vectors = []
            for i in range(len(self.calibration_data["marker_poses"])):
                # Get marker pose
                marker_pose = self.calibration_data["marker_poses"][i]
                T_marker2base = np.eye(4)
                T_marker2base[:3, :3] = Rotation.from_quat(
                    [
                        marker_pose["quaternion"][1],
                        marker_pose["quaternion"][2],
                        marker_pose["quaternion"][3],
                        marker_pose["quaternion"][0],
                    ]
                ).as_matrix()
                T_marker2base[:3, 3] = marker_pose["position"]
                T_cam2base = T_marker2base @ self.T_cam2marker
                T_base2cam = np.linalg.inv(T_cam2base)
                rvec = cv2.Rodrigues(T_base2cam[:3, :3])[0]
                tvec = T_base2cam[:3, 3]
                T_board2base = T_cam2base @ self.calibration_data["camera_poses"][i]
                R_board2base = T_board2base[:3, :3]
                t_board2base = T_board2base[:3, 3]
                rotation_matrices.append(R_board2base)
                translation_vectors.append(t_board2base)

            average_translation = np.mean(translation_vectors, axis=0)
            quaternions = [
                R.from_matrix(R_mat).as_quat() for R_mat in rotation_matrices
            ]

            average_quaternion = np.mean(quaternions, axis=0)
            average_quaternion /= np.linalg.norm(average_quaternion)  # Normalize

            average_rotation = R.from_quat(average_quaternion).as_matrix()

            T_board2base_avg = np.eye(4)
            T_board2base_avg[:3, :3] = average_rotation
            T_board2base_avg[:3, 3] = average_translation
            for i in range(len(self.calibration_data["marker_poses"])):
                # Get marker pose

                marker_pose = self.calibration_data["marker_poses"][i]
                T_marker2base = np.eye(4)
                T_marker2base[:3, :3] = Rotation.from_quat(
                    [
                        marker_pose["quaternion"][1],
                        marker_pose["quaternion"][2],
                        marker_pose["quaternion"][3],
                        marker_pose["quaternion"][0],
                    ]
                ).as_matrix()
                T_marker2base[:3, 3] = marker_pose["position"]
                T_cam2base = T_marker2base @ self.T_cam2marker
                T_base2cam = np.linalg.inv(T_cam2base)
                rvec = cv2.Rodrigues(T_base2cam[:3, :3])[0]
                tvec = T_base2cam[:3, 3]

                points_3d = self.calibration_data["object_points"][i]
                points_3d_homogeneous = np.hstack(
                    (
                        points_3d,
                        np.ones((points_3d.shape[0], 1)),
                    )
                )
                points_3d_base = T_board2base_avg @ points_3d_homogeneous.T
                points_3d_base = points_3d_base[:3, :]

                projected_points, _ = cv2.projectPoints(
                    points_3d_base,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                projected_points = projected_points.astype(img_points.dtype)

                projected_points_all.append(projected_points.reshape(-1, 2))
                ground_truth_points_all.append(
                    self.calibration_data["image_points"][i].reshape(-1, 2)
                )

                # Calculate error
                # error = cv2.norm(
                #     self.calibration_data["image_points"][i],
                #     projected_points,
                #     cv2.NORM_L2,
                # ) / len(projected_points)
                # breakpoint()
                image_points = self.calibration_data["image_points"][
                    i
                ].squeeze()  # Shape: (54, 2)
                projected_points = projected_points.squeeze()  # Shape: (54, 2)

                # Compute average point-wise L2 error
                error = np.linalg.norm(projected_points - image_points, axis=-1).mean()
                print(f"hand eye error for {i}th pair", error)
                hand_eye_errors.append(error)
        print("stats")
        # Compile statistics
        stats = {
            "camera_calibration": {
                "mean_error": mean_camera_error,
                "max_error": max(per_view_errors),
                "min_error": min(per_view_errors),
                "std_error": np.std(per_view_errors),
                "per_view_errors": per_view_errors,
            }
        }
        print("hand_eye_errors")
        if hand_eye_errors:
            stats["hand_eye"] = {
                "mean_error": np.mean(hand_eye_errors),
                "max_error": max(hand_eye_errors),
                "min_error": min(hand_eye_errors),
                "std_error": np.std(hand_eye_errors),
                "per_view_errors": hand_eye_errors,
            }
            print("hand_eye_errors stats")
            points_data = {
                "projected_points": [
                    points.tolist()
                    for points in projected_points_all  # Convert arrays to lists
                ],
                "ground_truth_points": [
                    points.tolist()
                    for points in ground_truth_points_all  # Convert arrays to lists
                ],
                "errors_per_frame": [
                    float(error) for error in hand_eye_errors
                ],  # Convert float32 to float
            }

            print("hand_eye_errors save stats")
            # Use the config object to save files
            points_filename = f"hand_eye_points_{base_filename}.json"
            with open(points_filename, "w") as f:
                json.dump(points_data, f, indent=4)
            print(f"\nSaved projected and ground truth points to {points_filename}")
        print("print")
        # Print detailed report
        print("\nReprojection Error Analysis:")
        print("\nCamera Calibration:")
        print(f"Mean error: {stats['camera_calibration']['mean_error']:.3f} pixels")
        print(f"Max error: {stats['camera_calibration']['max_error']:.3f} pixels")
        print(f"Min error: {stats['camera_calibration']['min_error']:.3f} pixels")
        print(f"Std error: {stats['camera_calibration']['std_error']:.3f} pixels")

        if "hand_eye" in stats:
            print("\nHand-Eye Calibration:")
            print(f"Mean error: {stats['hand_eye']['mean_error']:.3f} pixels")
            print(f"Max error: {stats['hand_eye']['max_error']:.3f} pixels")
            print(f"Min error: {stats['hand_eye']['min_error']:.3f} pixels")
            print(f"Std error: {stats['hand_eye']['std_error']:.3f} pixels")

        return stats, points_3d_base

    def visualize_with_stats(
        self, points_3d_base, image_index=0, image_path=None, stats=None, save_path=None
    ):
        """
        Visualize reprojection errors using comprehensive statistics

        Args:
            image_index: Index of the image to visualize
            stats: Statistics from calculate_reprojection_errors()
            save_path: Optional path to save the visualization
        """
        if stats is None:
            stats = self.calculate_reprojection_errors()

        camera_matrix = np.array(self.calibration_data["camera_matrix"])
        dist_coeffs = np.array(self.calibration_data["dist_coeffs"])

        # camera_matrix = np.array([[1305.5, 0, 599.1], [0, 1325.2, 508.5], [0, 0, 1]])
        # dist_coeffs = np.array([-0.2720, 0.0549, 0.0128, 0.0145, 0.1170])

        # Get original points and prepare for projection
        obj_points = self.calibration_data["object_points"][image_index]
        img_points = self.calibration_data["image_points"][image_index]

        # 1. Camera calibration projection
        rvec_cam = cv2.Rodrigues(
            np.array(self.calibration_data["camera_poses"][image_index])[:3, :3]
        )[0]
        tvec_cam = np.array(self.calibration_data["camera_poses"][image_index])[:3, 3]
        projected_points_cam, _ = cv2.projectPoints(
            obj_points, rvec_cam, tvec_cam, camera_matrix, dist_coeffs
        )

        # 2. Hand-eye calibration projection
        marker_pose = self.calibration_data["marker_poses"][image_index]
        T_marker2base = np.eye(4)
        T_marker2base[:3, :3] = Rotation.from_quat(
            [
                marker_pose["quaternion"][1],
                marker_pose["quaternion"][2],
                marker_pose["quaternion"][3],
                marker_pose["quaternion"][0],
            ]
        ).as_matrix()
        T_marker2base[:3, 3] = marker_pose["position"]

        # Calculate camera pose using hand-eye transformation
        #     T_cam2marker = np.array([[  -0.422618,    0.      ,   -0.906308, -274.799   ],
        #    [  -0.906308,    0.      ,   -0.422618, -153.868   ],
        #    [   0.      ,    1.      ,    0.      ,   20.8     ],
        #    [   0.      ,    0.      ,    0.      ,    1.      ]])
        # T_cam2marker=np.array([[ 0.4028,  -0.0476,  -0.9141,  -0.2652],
        #                         [ 0.9123,  -0.0598,   0.4051,   0.1610],
        #                         [-0.0739,  -0.9971,   0.0193,  -0.0165],
        #                         [ 0.0,      0.0,      0.0,      1.0000]])
        # T_cam2marker=np.array([[ 0.4028,  -0.0476,  -0.9141,  -265.2],
        #                         [ 0.9123,  -0.0598,   0.4051,   161.0],
        #                         [-0.0739,  -0.9971,   0.0193,  16.5],
        #                         [ 0.0,      0.0,      0.0,      1.0000]])
        T_cam2base = T_marker2base @ self.T_cam2marker
        T_base2cam = np.linalg.inv(T_cam2base)

        rvec = cv2.Rodrigues(T_base2cam[:3, :3])[0]
        tvec = T_base2cam[:3, 3]

        projected_points, _ = cv2.projectPoints(
            points_3d_base,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )

        # Load and create visualization
        img_path = f"{image_path}{sorted(os.listdir(image_path))[image_index]}"
        img = cv2.imread(img_path)

        # Create two copies of the image for side-by-side comparison
        h, w = img.shape[:2]
        vis_img = np.zeros(
            (h + 200, w * 2 + 20, 3), dtype=np.uint8
        )  # Added extra height for stats
        vis_img[:h, :w] = img.copy()
        vis_img[:h, w + 20 :] = img.copy()

        # Draw points and errors on both images
        # Left image: Camera calibration
        current_cam_error = stats["camera_calibration"]["per_view_errors"][image_index]
        for orig, proj in zip(img_points, projected_points_cam):
            pt_orig = tuple(orig[0].astype(int))
            pt_proj = tuple(proj[0].astype(int))

            # Color based on error magnitude (green to red)
            error = np.sqrt(np.sum((orig - proj) ** 2))
            error_ratio = error / stats["camera_calibration"]["max_error"]
            color = (0, int(255 * (1 - error_ratio)), int(255 * error_ratio))

            cv2.circle(vis_img, pt_orig, 3, (0, 255, 0), -1)
            cv2.circle(vis_img, pt_proj, 3, color, -1)
            cv2.line(vis_img, pt_orig, pt_proj, color, 1)

        # Right image: Hand-eye calibration
        current_he_error = stats["hand_eye"]["per_view_errors"][image_index]
        for orig, proj in zip(img_points, projected_points):
            pt_orig = tuple(orig[0].astype(int))
            pt_proj = tuple(proj[0].astype(int))
            pt_proj = (pt_proj[0] + w + 20, pt_proj[1])
            pt_orig = (pt_orig[0] + w + 20, pt_orig[1])

            error = np.sqrt(np.sum((orig - proj) ** 2))
            error_ratio = error / stats["hand_eye"]["max_error"]
            color = (0, 0, 255)

            cv2.circle(vis_img, pt_orig, 3, (0, 255, 0), -1)
            cv2.circle(vis_img, pt_proj, 3, color, -1)
            cv2.line(vis_img, pt_orig, pt_proj, color, 1)

        # Add detailed statistics
        stats_y = h + 30
        # Camera calibration stats
        cv2.putText(
            vis_img,
            "Camera Calibration Stats:",
            (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Current frame error: {current_cam_error:.3f}px",
            (10, stats_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Mean error: {stats['camera_calibration']['mean_error']:.3f}px",
            (10, stats_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Max error: {stats['camera_calibration']['max_error']:.3f}px",
            (10, stats_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Std error: {stats['camera_calibration']['std_error']:.3f}px",
            (10, stats_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Hand-eye calibration stats
        cv2.putText(
            vis_img,
            "Hand-Eye Calibration Stats:",
            (w + 30, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Current frame error: {current_he_error:.3f}px",
            (w + 30, stats_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Mean error: {stats['hand_eye']['mean_error']:.3f}px",
            (w + 30, stats_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Max error: {stats['hand_eye']['max_error']:.3f}px",
            (w + 30, stats_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_img,
            f"Std error: {stats['hand_eye']['std_error']:.3f}px",
            (w + 30, stats_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add frame counter
        cv2.putText(
            vis_img,
            f"Frame {image_index+1}/{len(stats['camera_calibration']['per_view_errors'])}",
            (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display or save the visualization
        if save_path:
            cv2.imwrite(save_path, vis_img)
        else:
            cv2.imshow("Calibration Errors Comparison", vis_img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            return key  # Return key press for navigation


def main():
    # Configuration
    experiment_id = "1122_1"  # Change this for different experiments
    method = "PARK"  # Change this for different methods

    # Create base filename
    base_filename = f"{experiment_id}_{method}"

    # Example usage
    collector = CalibrationDataCollector(checkerboard_size=(6, 9), square_size=5)

    # Load data from TrackerVisualizer's output directory
    data_path = Path(
        f"handEyeCalib_data/recorded_data_{experiment_id}"
    )  # Directory name also uses experiment_id
    collector.load_data_from_visualizer(data_path)
    print("calib data", len(collector.calibration_data["marker_poses"]))
    image_path = f"handEyeCalib_data/recorded_data_{experiment_id}/img/"
    # Perform camera calibration
    first_image = cv2.imread(f"{image_path}1732336597175.png")
    if first_image is not None:
        image_size = (first_image.shape[1], first_image.shape[0])
        print(image_size)
        success = collector.calibrate_camera(image_size)

        if success:
            print("Camera calibration successful!")
            # breakpoint()

            # Solve hand-eye calibration
            try:
                T_cam2marker = collector.solve_hand_eye_calibration()
                print("\nMarker to Camera Transformation Matrix:")
                print(T_cam2marker)

                # Save results using base filename
                try:
                    np.save(f"T_cam2marker_{base_filename}.npy", T_cam2marker)
                except Exception as save_matrix_error:
                    print(f"Failed to save transformation matrix: {save_matrix_error}")

                try:
                    error_stats, points_3d_base = (
                        collector.calculate_reprojection_errors(base_filename)
                    )

                    # Save error statistics
                    # try:
                    #     with open(f"error_stats_{base_filename}.txt", "w") as f:
                    #         json.dump(error_stats, f, indent=4)
                    # except Exception as save_stats_error:
                    #     print(f"Failed to save error statistics: {save_stats_error}")

                    # Visualize errors
                    os.makedirs(
                        f"reprojection_errors/{base_filename}_norm", exist_ok=True
                    )
                    for i in range(17):
                        try:
                            collector.visualize_with_stats(
                                points_3d_base,
                                i,
                                image_path,
                                error_stats,
                                f"reprojection_errors/{base_filename}_norm/{i}.png",
                            )
                        except Exception as visualization_error:
                            print(
                                f"Failed to visualize error for image {i}: {visualization_error}"
                            )

                except Exception as error_calculation_error:
                    print(
                        f"Failed to calculate reprojection errors: {error_calculation_error}"
                    )

            except Exception as calibration_error:
                print(f"Hand-eye calibration failed: {calibration_error}")
                # Optionally, log more detailed error information
                import traceback

                traceback.print_exc()
        else:
            print("Camera calibration failed!")
    else:
        print("Error: Could not load any images for calibration.")


if __name__ == "__main__":
    main()
