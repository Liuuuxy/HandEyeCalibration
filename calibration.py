import cv2
import numpy as np
import glob
import logging
from datetime import datetime

def main():
    # Define the chess board rows and columns
    rows = 6
    cols = 9
    square_size_mm = 5  # Size of each square in millimeters

    # Set termination criteria for the iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points with actual physical dimensions
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size_mm  # Convert to millimeters

    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space (mm)
    imgpoints = []  # 2d points in image plane (pixels)

    # Load images
    images = glob.glob("calibration_images/1108/*.jpg")
    print(f"Found {len(images)} images")

    # Get size from first image
    first_image = cv2.imread(images[0])
    height, width = first_image.shape[:2]
    print(f"Using image size: {width}x{height}")

    for fname in images:
        img = cv2.imread(fname)
        if img.shape[:2] != (height, width):
            print(f"Warning: Image {fname} has different size. Skipping.")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )

    print("\nCamera Calibration Results:")
    print(f"Image size: {width}x{height}")
    print("\nCamera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"\nTotal reprojection error: {mean_error/len(objpoints)}")

    # Save results
    np.save('camera_matrix.npy', mtx)
    np.save('dist_coeffs.npy', dist)
    
    # Save calibration info to text file
    with open('calibration_info.txt', 'w') as f:
        f.write(f"Camera Resolution: {width}x{height}\n")
        f.write(f"Calibration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of images used: {len(images)}\n")
        f.write(f"Checkerboard size: {cols}x{rows}\n")
        f.write(f"Square size: {square_size_mm}mm\n")
        f.write(f"\nCamera Matrix:\n{str(mtx)}\n")
        f.write(f"\nDistortion Coefficients:\n{str(dist)}\n")
        f.write(f"\nReprojection Error: {mean_error/len(objpoints)}\n")

if __name__ == "__main__":
    main()
