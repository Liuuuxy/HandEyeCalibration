# Hand-Eye Calibration System

A system for performing hand-eye calibration using a checkerboard pattern and NDI Polaris tracking system.

## Overview

This project implements a hand-eye calibration system that:
1. Records synchronized data from an NDI Polaris tracker and a camera
2. Performs camera calibration using a checkerboard pattern
3. Computes hand-eye calibration using various methods (PARK, ANDREFF)
4. Evaluates calibration quality through reprojection error analysis

## System Requirements

- Python 3.8+
- OpenCV
- NumPy
- NDI Polaris tracking system
- USB camera
- Checkerboard pattern (6x9 squares, 5mm square size)
- NDI marker ROM files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HandEyeCalibration.git
cd HandEyeCalibration
```

2. Install dependencies:
```bash
pip install opencv-python numpy scipy scikit-surgerynd
```

## Project Structure

```
HandEyeCalibration/
├── recorder.py           # Data recording from camera and tracker
├── calibration.py        # Camera calibration script
├── hand_eye_calib.py     # Hand-eye calibration implementation
├── project_target.py     # Target projection verification
└── recorded_data_*/      # Recorded data directories
    ├── tracking_data.json
    └── img/
```

## Usage

### 1. Data Recording

Run the recorder to collect synchronized camera and tracker data:

```bash
python recorder.py
```
- Press SPACE to capture a frame when all markers are detected
- Press ESC to finish recording

### 2. Camera Calibration

Calibrate the camera using the checkerboard images:

```bash
python calibration.py
```

This generates:
- `camera_matrix.npy`
- `dist_coeffs.npy`
- `calibration_info.txt`

### 3. Hand-Eye Calibration

Perform hand-eye calibration:

```bash
python hand_eye_calib.py
```

Outputs:
- Transformation matrix (`T_marker2cam_*.npy`)
- Error statistics (`error_stats_*.txt`)
- Reprojection visualizations

## Configuration

### Tracker Settings
```python
settings = {
    "tracker type": "polaris",
    "ip address": "192.168.2.17",
    "port": 8765,
    "romfiles": ["path/to/rom/file"],
    "use quaternions": True
}
```

### Calibration Parameters
- Checkerboard: 6x9 squares
- Square size: 5mm
- Camera resolution: 1280x720 (falls back to 640x480)

## File Naming Convention

Files are named using the pattern:
`{experiment_id}_{method}`

Example:
- `T_marker2cam_1112_2_PARK.npy`
- `error_stats_1112_2_PARK.txt`

## Results

The system provides:
1. Camera calibration parameters
2. Hand-eye transformation matrix
3. Reprojection error analysis
4. Visualization of calibration quality

