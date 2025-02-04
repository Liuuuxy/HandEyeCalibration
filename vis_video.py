import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import json


def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a rotation matrix.
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_matrix()


def get_transformation_matrix(position, quaternion):
    """
    Create a 4x4 transformation matrix from position and quaternion.
    """
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix


json_path = "handEyeCalib_data\\recorded_data_0127_cam_1\\tracking_data.json"
with open(json_path, "r") as f:
    tracking_data = json.load(f)

video_path = "handEyeCalib_data\\recorded_data_0127_cam_1\\video\\video_1738015374.mp4"
cap = cv2.VideoCapture(video_path)

K = np.array(
    # [
    #     [1.32105991e03, 0.00000000e00, 7.11598371e02],
    #     [0.00000000e00, 1.31810972e03, 3.73443609e02],
    #     [0.00000000e00, 0.00000000e00, 1.00000000e00],
    # ]
    [
        [1.31108397e03, 0.00000000e00, 6.86486022e02],
        [0.00000000e00, 1.31801722e03, 4.41608492e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# dist_coeffs = np.array([-0.23051146, 0.18608859, 0.00667657, 0.0133185, -0.90655494])

dist_coeffs = np.array([-0.21975247, -0.01772734, 0.00826037, -0.00210365, -0.24672817])

# Iterate through frames and markers
output_path = (
    "handEyeCalib_data/recorded_data_0127_cam_1/video/marker_path_overlay_test_0203.mp4"
)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
marker_paths = []

frame_keys = list(tracking_data["1738015374"]["frames"].keys())

T_marker1_to_camera = np.array(
    # [
    #     [3.96653926e-01, 1.13394080e-01, -9.10937674e-01, -2.65475128e02],
    #     [9.17873600e-01, -6.32405663e-02, 3.91801844e-01, 1.47919302e02],
    #     [-1.31802046e-02, -9.91535382e-01, -1.29166052e-01, -2.05290873e01],
    #     [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    # ]
    [
        [4.45038364e-01, 1.01815503e-01, -8.89704703e-01, -2.65384336e02],
        [8.94482122e-01, -2.91752819e-03, 4.47094198e-01, 1.48419097e02],
        [4.29253819e-02, -9.94799021e-01, -9.23705588e-02, -1.88705508e01],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# polais is also in mm, so no need to convert for this recorded data
# T_marker1_to_camera[:3, 3] = T_marker1_to_camera[:3, 3] / 1000

paths = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the frame for the faded path overlay
    overlay = np.zeros_like(frame)

    frame_key = frame_keys[frame_idx]
    frame_data = tracking_data["1738015374"]["frames"][frame_key]["markers"]

    if not frame_data[0]:
        # Write the processed frame to the output video
        out.write(frame)
        frame_idx += 1
        continue
    marker1 = frame_data[0]["quaternion"][0]

    T_polaris_to_marker1 = get_transformation_matrix(marker1[4:7], marker1[:4])
    T_polaris_to_camera = T_polaris_to_marker1 @ T_marker1_to_camera
    T_camera_to_polaris = np.linalg.inv(T_polaris_to_camera)

    if frame_data[1]:
        marker2 = frame_data[1]["quaternion"][0]
        T_polaris_to_marker2 = get_transformation_matrix(marker2[4:7], marker2[:4])
        paths.append(T_polaris_to_marker2)
    if len(paths) > 100:  # Keep the last 30 points
        paths = paths[-100:]

    marker_paths = []

    # Draw fading path
    for idx, T_polaris_to_marker2 in enumerate(paths):

        # Extract rotation and translation from the transformation matrix
        rotation_matrix_camera_to_polaris = T_camera_to_polaris[:3, :3]
        position_camera_to_polaris = T_camera_to_polaris[:3, 3]

        # Convert rotation matrix to rotation vector (Rodrigues)
        rvec, _ = cv2.Rodrigues(rotation_matrix_camera_to_polaris)
        tvec = position_camera_to_polaris.reshape(3, 1)

        point_3d = np.array(T_polaris_to_marker2[:3, 3], dtype=np.float32)

        uv_distorted, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist_coeffs)

        # Extract the distorted coordinates and convert to integers
        u, v = uv_distorted.ravel()

        u, v = int(round(u)), int(round(v))
        print(f"Marker position: ({u}, {v})")

        marker_paths.append((u, v))

    PATH_COLOR = (0, 255, 0)  # Green

    # Define maximum opacity for the path
    MAX_OPACITY = 0.7  # Adjust as needed for visibility

    # Draw fading path
    for j in range(1, len(marker_paths)):
        # Calculate opacity based on position in path (older points are more transparent)
        alpha = (j / len(marker_paths)) * MAX_OPACITY  # Newer points are more opaque

        # Create a temporary overlay for the current line
        temp_overlay = np.zeros_like(frame)

        # Draw the line segment on the temporary overlay
        cv2.line(
            temp_overlay,
            marker_paths[j - 1],
            marker_paths[j],
            PATH_COLOR,
            thickness=3,  # Increased thickness for better visibility
        )

        # Blend the temporary overlay with the main overlay using the calculated alpha
        overlay = cv2.addWeighted(temp_overlay, alpha, overlay, 1.0, 0)

    # Blend the overlay with the frame
    frame = cv2.addWeighted(overlay, 1.0, frame, 1.0, 0)

    # Draw current marker position (solid green dot)
    if marker_paths:
        cv2.circle(frame, marker_paths[-1], 5, (0, 255, 0), -1)

    # Write the processed frame to the output video
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# Save the visualization
print(f"Visualization saved at {output_path}")
