import numpy as np

def rotate_pose(poses, angles_degrees, axis='y'):
    # Convert angles to radians
    angles_radians = np.radians(angles_degrees)

    # Number of poses in the batch
    b = poses.shape[0]
    rotation_matrix = np.zeros((3, 3, b))
    # Initialize rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix[0, 0, :] = 1
        rotation_matrix[1, 1, :] = np.cos(angles_radians)
        rotation_matrix[1, 2, :] = -np.sin(angles_radians)
        rotation_matrix[2, 1, :] = np.sin(angles_radians)
        rotation_matrix[2, 2, :] = np.cos(angles_radians)
    elif axis == 'y':
        rotation_matrix[0, 0, :] = np.cos(angles_radians)
        rotation_matrix[0, 2, :] = np.sin(angles_radians)
        rotation_matrix[1, 1, :] = 1
        rotation_matrix[2, 0, :] = -np.sin(angles_radians)
        rotation_matrix[2, 2, :] = np.cos(angles_radians)
    elif axis == 'z':
        rotation_matrix[0, 0, :] = np.cos(angles_radians)
        rotation_matrix[0, 1, :] = -np.sin(angles_radians)
        rotation_matrix[1, 0, :] = np.sin(angles_radians)
        rotation_matrix[1, 1, :] = np.cos(angles_radians)
        rotation_matrix[2, 2, :] = 1
    else:
        raise ValueError("Invalid axis. Please choose 'x', 'y', or 'z'.")
    # Rotate the poses
    rotated_poses = np.matmul(poses, rotation_matrix.transpose((2,1,0)))

    return rotated_poses