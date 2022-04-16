"""
KITTI format to YOLOv5 format
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

"""
KITTI format 2D Bbox
[x1, y1, x2, y2]
[left_top, bottom_right]

YOLO Format
[x_center, y_center, width, height]
"""

"""
#Values    Name      Description
----------------------------------------------------------------------------
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.
"""

class KITTIYOLO:
    def __init__(self, labels_path, output_path):
        self.labels_path = labels_path
        self.output_path = output_path

        # kitti class
        self.ids = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Truck': 3
        }

        # ignored class
        self.ignore_class = {
            "Van", 
            "Tram", 
            "Person_sitting", 
            "DontCare", 
            "Misc"}

        # image information
        self.im_width = 1224
        self.im_height = 370

    def parse_line(self, line):
        parts = line.split(" ")
        output = {
            "name": parts[0].strip(),
            "xyz_camera": (float(parts[11]), float(parts[12]), float(parts[13])),
            "wlh": (float(parts[9]), float(parts[10]), float(parts[8])),
            "yaw_camera": float(parts[14]),
            "bbox_camera": (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
            "truncation": float(parts[1]),
            "occlusion": float(parts[2]),
            "alpha": float(parts[3]),
        }

        # Add score if specified
        if len(parts) > 15:
            output["score"] = float(parts[15])
        else:
            output["score"] = np.nan

        return output

    def kitti_to_yolo(self):
        files = glob(self.labels_path + '/*.txt')
        for file in tqdm(files):
            with open(file, 'r') as f:
                fn = os.path.join(self.output_path, file.split('/')[-1])
                dump_txt = open(fn, 'w')
                for line in f:
                    parsed_line = self.parse_line(line)
                    if parsed_line["name"] in self.ignored_class:
                        continue

                    xmin, ymin, xmax, ymax = parsed_line['bbox_camera']
                    xcenter = ((xmax - xmin)/2 + xmin) / self.im_width
                    ycenter = ((ymax - ymin)/2 + ymin) / self.im_height
                    width = (xmax - xmin) / self.im_width
                    height = (ymax - ymin) / self.m_height
                    
                    bbox_yolo = f"{self.ids[parsed_line['name']]} {xcenter:.3f} {ycenter:.3f} {width:.3f} {height:.3f}"
                    dump_txt.write(bbox_yolo + "\n")
                    
                dump_txt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO labels')
    parser.add_argument('--labels_path', type=str, default='KITTI/label_2/')
    parser.add_argument('--output_path', type=str, default='KITTI/labels/')

    args = parser.parse_args()

    converter = KITTIYOLO(args.labels_path, args.output_path)
    converter.kitti_to_yolo()
