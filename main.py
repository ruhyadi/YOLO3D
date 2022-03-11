import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import shutil
from pathlib import Path
import sys
import argparse

from inference import detect3d

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    FILENAME = {}
    # get file 
    image = request.files['image']

    # save file
    image.save('static/image_eval.png')

    if 'image' in request.files:
        detect = True

        # process file
        detect3d(
            reg_weights='weights/epoch_10.pkl',
            model_select='resnet',
            source='static',
            calib_file='eval/camera_cal/calib_cam_to_cam.txt',
            save_result=True,
            show_result=False,
            output_path='static/'
        )

        # encode to base64 image
        with open('static/000.png', 'rb') as image_file:
            img_encode = base64.b64encode(image_file.read())
            to_send = 'data:image/png;base64, ' + str(img_encode, 'utf-8')
    else:
        detect = False

    return render_template('index.html', init=True, detect=detect, image_to_show=to_send)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


if __name__ == '__main__':
    try:
        os.makedirs('static')
    except:
        print('Directory already exist!')
    opt = parse_opt()
    app.run(debug=True, host='0.0.0.0', port=5020)
    shutil.rmtree('static')