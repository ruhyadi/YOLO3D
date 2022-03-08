"""
Script for handling calibration file
"""

import numpy as np

def get_P(calib_file):
    """
    Get matrix P_rect_02 (camera 2 RGB)
    and transform to 3 x 4 matrix
    """
    for line in open(calib_file):
        if 'P_rect_02' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            matrix = np.zeros((3, 4))
            matrix = cam_P.reshape((3, 4))
            return matrix

# TODO: understand this

def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

    file_not_found(cab_f)

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))

            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0

            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,:4] = Tr

            return Tr_to_velo

def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()

