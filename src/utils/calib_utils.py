"""Camera calibration utils."""

import rootutils

ROOT = rootutils.autosetup()

import numpy as np


def load_matrix_p2(calib_path: str) -> np.ndarray:
    """
    Get matrx P_rect_02 (camera 2 RGB) then transform to 3 x 4 matrix.

    Args:
        calib_path (str): path to calibration file.

    Returns:
        np.ndarray: 3 x 4 matrix.

    Examples:
        >>> calib_path = "./data/KITTI/calibrations.txt"
        >>> print(get_P(calib_path))
            [[ 7.188560e+02  0.000000e+00  6.071928e+02  4.538225e+01]
            [ 0.000000e+00  7.188560e+02  1.852157e+02 -1.130887e-01]
            [ 0.000000e+00  0.000000e+00  1.000000e+00  3.779761e-03]]
    """
    for line in open(calib_path, "r"):
        if "P_rect_02" in line:
            cam_P = line.strip().split(" ")
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            matrix = np.zeros((3, 4))
            matrix = cam_P.reshape((3, 4))

    return matrix


def load_matrix_r0(calib_path: str) -> np.ndarray:
    """
    Get matrix R0_rect then transform to 3 x 3 matrix.

    Args:
        calib_path (str): path to calibration file.

    Returns:
        np.ndarray: 3 x 3 matrix.
    """
    for line in open(calib_path, "r"):
        if "R0_rect:" in line:
            cam_R = line.strip().split(" ")
            cam_R = np.asarray([float(cam_R) for cam_R in cam_R[1:]])
            cam_R = cam_R.reshape((3, 3))
            matrix_R0_rect = np.zeros([4, 4])
            matrix_R0_rect[3, 3] = 1
            matrix_R0_rect[:3, :3] = cam_R

    return matrix_R0_rect


def load_matrix_tr(calib_path: str) -> np.ndarray:
    """
    Get matrix Tr_velo_to_cam then transform to 3 x 4 matrix.

    Args:
        calib_path (str): path to calibration file.

    Returns:
        np.ndarray: 3 x 4 matrix.
    """
    for line in open(calib_path, "r"):
        if "Tr_velo_to_cam:" in line:
            cam_T = line.strip().split(" ")
            cam_T = np.asarray([float(cam_T) for cam_T in cam_T[1:]])
            cam_T = cam_T.reshape((3, 4))
            matrix_Tr_velo_to_cam = np.zeros([4, 4])
            matrix_Tr_velo_to_cam[3, 3] = 1
            matrix_Tr_velo_to_cam[:3, :] = cam_T

    return matrix_Tr_velo_to_cam
