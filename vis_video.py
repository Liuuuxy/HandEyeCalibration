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


def rotation_matrix_to_quaternion(matrix):
    """
    Convert a rotation matrix into a quaternion.
    """
    r = R.from_matrix(matrix)
    return r.as_quat()


def extract_position_and_quaternion(transformation_matrix):
    """
    Extract position and quaternion from a 4x4 transformation matrix.
    """
    position = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return position, quaternion


json_path = "handEyeCalib_data\\recorded_data_1122_2\\tracking_data.json"
with open(json_path, "r") as f:
    tracking_data = json.load(f)

video_path = "handEyeCalib_data/recorded_data_1122_2/video/video_1736362354.mp4"
cap = cv2.VideoCapture(video_path)

K = np.array(
    [
        [1.30832797e03, 0.00000000e00, 5.96257634e02],
        [0.00000000e00, 1.32862288e03, 5.09622317e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
dist_coeffs = np.array([-0.28331112, 0.17623822, 0.01275944, 0.01448258, -0.29583552])

camera_to_base_rvec = np.zeros((3, 1))
camera_to_base_tvec = np.zeros((3, 1))

# Iterate through frames and markers
output_path = (
    "handEyeCalib_data/recorded_data_1122_2/video/marker_path_overlay_fixed.mp4"
)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
marker_paths = []

frame_keys = list(tracking_data["1736362354"]["frames"].keys())

T_camera_to_marker1 = np.array(
    [
        [3.96653926e-01, 1.13394080e-01, -9.10937674e-01, -2.65475128e02],
        [9.17873600e-01, -6.32405663e-02, 3.91801844e-01, 1.47919302e02],
        [-1.31802046e-02, -9.91535382e-01, -1.29166052e-01, -2.05290873e01],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# polais is also in mm, so no need to convert
# T_camera_to_marker1[:3, 3] = T_camera_to_marker1[:3, 3] / 1000

paths = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the frame for the faded path overlay
    overlay = np.zeros_like(frame)

    frame_key = frame_keys[frame_idx]
    frame_data = tracking_data["1736362354"]["frames"][frame_key]["markers"]

    marker1 = frame_data[0]["quaternion"][0]

    T_marker1_to_polaris = get_transformation_matrix(marker1[4:7], marker1[:4])
    T_camera_to_polaris = T_marker1_to_polaris @ T_camera_to_marker1
    T_polaris_to_camera = np.linalg.inv(T_camera_to_polaris)

    if frame_data[1]:
        marker2 = frame_data[1]["quaternion"][0]
        T_marker2_to_polaris = get_transformation_matrix(marker2[4:7], marker2[:4])
        paths.append(T_marker2_to_polaris)
    if len(paths) > 30:  # Keep the last 30 points
        paths = paths[-30:]

    marker_paths = []

    # Draw fading path
    for idx, T_marker2_to_polaris in enumerate(paths):
        T_marker2_to_camera = T_polaris_to_camera @ T_marker2_to_polaris

        # Extract rotation and translation from the transformation matrix
        rotation_matrix_marker2_to_camera = T_marker2_to_camera[:3, :3]
        position_marker2_to_camera = T_marker2_to_camera[:3, 3]

        # Convert rotation matrix to rotation vector (Rodrigues)
        rvec, _ = cv2.Rodrigues(rotation_matrix_marker2_to_camera)
        tvec = position_marker2_to_camera.reshape(3, 1)

        point_3d = np.array([[0, 0, 0]], dtype=np.float32)

        uv_distorted, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist_coeffs)

        # Extract the distorted coordinates and convert to integers
        u, v = uv_distorted.ravel()
        # v = frame.shape[0] - v - 300
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
