import numpy as np
import cv2
import os


def get_P():
    # Intrinsic parameters
    fx_l = 710.0990  # focal length in x-direction
    fy_l = 710.5100    # focal length in y-direction
    cx_l = 636.4620   # principal point x-coordinate
    cy_l = 346.3210  # principal point y-coordinate

    fx_r = 706.1510
    fy_r = 706.5660
    cx_r = 624.5390
    cy_r = 333.2290

    d_l = np.array([-0.1711, 0.0219,  0.0006, -0.0007, 0.0024])
    d_r = np.array([-0.1741, 0.0248,  0.0006,  -0.0007, 0.0024])
    # Extrinsic parameters (example values)
    # Assuming rotation matrix R and translation vector t are already defined

    # Rotation angles around X, Y, and Z axes (in radians)
    Rx = 0.0161  # Rotation around X-axis
    Ry = -0.0006     # Rotation around Y-axis
    Rz = -0.0030   # Rotation around Z-axis

    # Rotation matrices around X, Y, and Z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(Rx), -np.sin(Rx)],
                    [0, np.sin(Rx), np.cos(Rx)]])

    R_y = np.array([[np.cos(Ry), 0, np.sin(Ry)],
                    [0, 1, 0],
                    [-np.sin(Ry), 0, np.cos(Ry)]])

    R_z = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                    [np.sin(Rz), np.cos(Rz), 0],
                    [0, 0, 1]])

    # Combine the rotation matrices to get the final rotation matrix R
    R = np.dot(R_z, np.dot(R_y, R_x))
    t = np.array([119.9520, 0, 0])  # Example translation vector

    # Construct the intrinsic matrix K
    K_left = np.array([[fx_l, 0, cx_l],
                [0, fy_l, cy_l],
                [0, 0, 1]])
    K_right = np.array([[fx_r, 0, cx_r],
                        [0, fy_r, cy_r],
                        [0, 0, 1]])
    # Rectify images
    R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(K_left,d_l ,K_right, d_r, (1280, 720), R, t)
    print("Rotation_left:")
    print(R_left)
    print("Rotation right:")
    print(R_right)
    print("Projection matrix for left cam using Open CV:")
    print(P_left)
    print("Projection matrix for right cam using Open CV:")
    print(P_right)
    # Combine rotation matrix R and translation vector t
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))  # [R | t]

    # Construct the projection matrix P
    P_left = np.dot(K_left, extrinsic_matrix)
    P_right = np.dot(K_right, extrinsic_matrix)
    print("Projection Matrix for left camera using K|[R|t]:")
    print(P_left)
    print("Projection matrix for right cam using K|[R|t]:")
    print(P_right)

def get_dim():
    # Directory containing the images
    directory = '/home/badri/IV_SLAM/Jackal_Visual_Odom/sequences/00033/image_1'

    # Get list of image files in the directory
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Loop through each image file
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read image '{image_file}'")
        else:
            # Get dimensions of the image
            height, width, channels = image.shape
            print(f"Image: {image_file}, Dimensions: {width}x{height}, Channels: {channels}")

def change_name_of_images():
    # Directory containing the images
    directory = '/home/badri/IV_SLAM/Jackal_Visual_Odom/sequences/00033/image_1'

    # Get list of image files in the directory
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Sort the image files to maintain order
    image_files.sort()

    # Counter for renaming
    counter = 0

    # Loop through each image file
    for image_file in image_files:
        # Rename the image file
        old_path = os.path.join(directory, image_file)
        new_filename = f"{counter:06d}.png"
        new_path = os.path.join(directory, new_filename)  # Format the new filename
        os.rename(old_path, new_path)
        
        # Print message
        print(f"Renamed {image_file} to {new_filename}")
        # Increment counter
        counter += 1  

    print("Renaming complete.")


if __name__ == "__main__":
    #get_P()
    change_name_of_images()
    #get_dim()