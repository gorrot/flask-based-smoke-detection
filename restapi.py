# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
from datetime import datetime, timedelta

import torch
from flask import Flask, request, render_template, Response, session, jsonify
from PIL import Image

from flask import Flask, request
import numpy as np
import cv2
import torch
import pandas
from pprint import pprint
from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
import logging
# from logging import FileHandler
# from logging.handlers import TimedRotatingFileHandler
from log_config import setup_logger, log_detection

app = Flask(__name__)

user, pwd, ip = "admin", "123456zaQ", "[192.168.100.196]"
model = None
INPUT = 'smoke_viedo/Black_smoke_517.avi'
tmp = None
alarm_time = datetime.now() - timedelta(seconds=5)
interval_seconds = 10
alarm_status = False
detection_paused = False

logger = setup_logger()


class CameraCapture:
    def __init__(self):
        self.ret = True
        self.INPUT = INPUT
        self.capture_img = None
        imgsz = 640
        # self.model = torch.hub.load('./', 'custom', 'runs/train/exp21/weights/best.pt', source='local')
        # self.device = select_device('')
        # self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        # self.stride = int(self.model.stride.max())  # model stride
        # self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        # 通过opencv获取实时视频流（海康摄像头）
        # self.video = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, 1))
        # 大华摄像头
        # self.video = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
        # set_logging()


def gen(inputx):
    cap = cv2.VideoCapture(inputx)
    while cap.isOpened():
        for i in range(1):
            ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)
            result_img = results.render()[0]
            result_img = cv2.resize(result_img, (600, 500))
            result_img = cv2.imencode('.jpg', result_img)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + result_img + b'\r\n')


def camera_cap(inputx):
    cap = cv2.VideoCapture(inputx)
    num_detection = 0
    global detection_paused
    while cap.isOpened():
        for i in range(1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not detection_paused:
                    results = model(frame)
                    num_detection = len(results.xyxy[0])
                    if num_detection > 0:
                        global alarm_time, interval_seconds
                        detect_time = datetime.now()
                        interval_time = (detect_time - alarm_time).total_seconds()
                        alarm_time = detect_time

                        if interval_time < interval_seconds:
                            global alarm_status
                            alarm_status = True
                            log_detection(logger, num_detection, alarm_status)

                        result_img = results.render()[0]

                    else:
                        result_img = frame
                else:
                    result_img = frame
                result_img = cv2.resize(result_img, (600, 500))
                result_img = cv2.imencode('.jpg', result_img)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + result_img + b'\r\n')


@app.before_first_request
def load_model():
    global model

    model = torch.hub.load('./', 'custom', 'runs/train/bi+wise/weights/best.pt', source='local')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/alarm_act/', methods=['GET', "POST"])
def alarm_act():
    global alarm_status
    data = {
        'js_alarm': alarm_status,
    }

    return jsonify(data)


@app.route('/alarm_dis/', methods=['GET', "POST"], strict_slashes=False)
def alarm_dis():
    global alarm_status
    alarm_status = request.form.get('js_alarm')
    data = {
        'js_alarm': alarm_status,
    }
    return jsonify(data)


@app.route('/video_feed', methods=['GET', "POST"])
def video_feed():
    return Response(gen(CameraCapture().INPUT), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_feed', methods=['GET', "POST"])
def capture_feed():
    return Response(camera_cap(CameraCapture().INPUT), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear_log', methods=['POST'])
def clear_log():
    # 清除之前的日志处理程序
    with open('runs/log/restapi.log', 'w'):
        pass
    app.logger.error('logs has been cleared')
    return jsonify({'status': 'success', 'message': '日志已清除'})


@app.route('/delay_detection', methods=['POST'])
def delay_detection():
    global detection_paused
    detection_paused = not detection_paused
    return jsonify({'status': 'success', 'js_detection_paused': detection_paused, 'message': '检测状态已切换'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
