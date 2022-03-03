"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""

from detect import run

from torch_lib.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet18

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

parser.add_argument("--video-dir", default="eval/video/",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        
        # TODO: Change model to Resnet
        my_resnet = resnet18(pretrained=True)
        my_resnet.fc = nn.Linear(512, 512)
        model = Model.Model_Resnet(model=my_resnet, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        image_dir = FLAGS.video_dir

    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()
        img_file = img_path + img_id + ".png"

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)

        # TODO: change to YOLOv5
        detections = run(source=img_file)

        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            # Using NN/VGG to predict orient, conf, and dim
            # TODO: Change model to Resnet
            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            if not FLAGS.hide_debug:
                print('Estimated pose: %s'%location)

        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imwrite(f'./with_yolo_{img_id}.png', numpy_vertical)
            #cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imwrite(f'./3D-detection_{img_id}.png', img)
            #cv2.imshow('3D detections', img)

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        # if FLAGS.video:
        #     cv2.waitKey(1)
        # else:
        #     if cv2.waitKey(0) != 32: # space bar
        #         exit()

if __name__ == '__main__':
    main()