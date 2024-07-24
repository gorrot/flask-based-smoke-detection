import argparse
import io
from datetime import datetime, timedelta
import torch
from flask import Flask, request, render_template, Response, jsonify, session
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from pprint import pprint
from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
import logging
from log_config import setup_logger, log_detection
import detect_api
import mix_detect

app = Flask(__name__)

# 配置全局变量
app.config['USER'] = "admin"
app.config['PWD'] = "123456zaQ"
app.config['IP'] = "[192.168.100.196]"
app.config['INPUT'] = 'smoke_viedo/Black_smoke_517.avi'
app.config['ALARM_TIME'] = datetime.now() - timedelta(seconds=5)
app.config['INTERVAL_SECONDS'] = 10
app.config['ALARM_STATUS'] = False
app.config['DETECTION_PAUSED'] = False
app.config['VIBE_PAUSED'] = False
app.config['MODEL'] = None

logger = setup_logger()


class CameraCapture:
    def __init__(self, input_source):
        self.ret = True
        self.INPUT = input_source
        self.capture_img = None
        imgsz = 640
        # Example of model loading and device selection
        # self.model = attempt_load(weights, map_location=select_device(''))
        # self.stride = int(self.model.stride.max())
        # self.imgsz = check_img_size(imgsz, s=self.stride)




def handle_both_paused(frame):
    result_img = cv2.resize(frame, (600, 500))  # 调整图像大小
    _, jpeg = cv2.imencode('.jpg', result_img)
    return jpeg.tobytes()


def handle_detection_only(frame):
    result_img = app.config['MODEL'](frame)
    result_img = result_img.render()[0]  # 确保渲染结果
    result_img = cv2.resize(result_img, (600, 500))  # 调整图像大小
    _, jpeg = cv2.imencode('.jpg', result_img)
    return jpeg.tobytes()


def handle_vibe_only(cap):
    frames = mix_detect.main(cap)
    for _, init_frame in frames:
        frame = cv2.resize(_, (600, 500))  # 调整图像大小
        _, jpeg = cv2.imencode('.jpg',frame)
        return jpeg.tobytes()


def handle_both_active(cap):
    frames = mix_detect.main(cap)
    for _, init_frame in frames:
        result_img = app.config['MODEL'](_)
        result_img = result_img.render()[0]  # 确保渲染结果
        result_img = cv2.resize(result_img, (600, 500))  # 调整图像大小
        _, jpeg = cv2.imencode('.jpg', result_img)
        return jpeg.tobytes()

def gen(inputx):
    cap = cv2.VideoCapture(inputx)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {inputx}")
        return  # 视频源无效时退出

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from video source")
            break

        try:
            if app.config['VIBE_PAUSED'] and app.config['DETECTION_PAUSED']:
                logger.debug("Both Vibe and Detection paused")
                frame = handle_both_paused(frame)
            elif app.config['VIBE_PAUSED']:
                logger.debug("Vibe paused, Detection not paused")
                frame = handle_detection_only(frame)
            elif app.config['DETECTION_PAUSED']:
                logger.debug("Detection paused, Vibe not paused")
                frame = handle_vibe_only(cap)
            else:
                logger.debug("Both Vibe and Detection are active")
                frame = handle_both_active(cap)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            continue


def camera_cap(inputx):
    cap = cv2.VideoCapture(inputx)
    num_detection = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not app.config['DETECTION_PAUSED']:
            results = app.config['MODEL'](frame)
            num_detection = len(results.xyxy[0])
            if num_detection > 0:
                detect_time = datetime.now()
                interval_time = (detect_time - app.config['ALARM_TIME']).total_seconds()
                app.config['ALARM_TIME'] = detect_time

                if interval_time < app.config['INTERVAL_SECONDS']:
                    app.config['ALARM_STATUS'] = True
                    log_detection(logger, num_detection, app.config['ALARM_STATUS'])

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
    app.config['MODEL'] = torch.hub.load('./', 'custom', 'runs/train/bi+wise/weights/best.pt', source='local')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/alarm_act', methods=['GET', "POST"])
def alarm_act():
    return jsonify({'js_alarm': app.config['ALARM_STATUS']})


@app.route('/alarm_dis', methods=['GET', "POST"], strict_slashes=False)
def alarm_dis():
    app.config['ALARM_STATUS'] = request.form.get('js_alarm')
    return jsonify({'js_alarm': app.config['ALARM_STATUS']})


@app.route('/video_feed', methods=['GET', "POST"])
def video_feed():
    return Response(gen(app.config['INPUT']), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_feed', methods=['GET', "POST"])
def capture_feed():
    return Response(camera_cap(app.config['INPUT']), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear_log', methods=['POST'])
def clear_log():
    with open('runs/log/restapi.log', 'w'):
        pass
    app.logger.error('logs has been cleared')
    return jsonify({'status': 'success', 'message': '日志已清除'})


@app.route('/delay_detection', methods=['POST'])
def delay_detection():
    app.config['DETECTION_PAUSED'] = not app.config['DETECTION_PAUSED']
    return jsonify({'status': 'success', 'js_detection_paused': app.config['DETECTION_PAUSED'], 'message': '检测状态已切换'})


@app.route('/Vibe_paused', methods=['POST'])
def Vibe_paused():
    app.config['VIBE_PAUSED'] = not app.config['VIBE_PAUSED']
    return jsonify({'status': 'success', 'js_vibe_paused': app.config['VIBE_PAUSED'], 'message': '检测状态已切换'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
