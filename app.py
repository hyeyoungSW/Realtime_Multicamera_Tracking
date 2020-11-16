from importlib import import_module
from flask import Flask, render_template, Response, jsonify
from flask import Blueprint, request, flash, redirect, url_for
from flask import current_app as current_app
import tensorflow as tf
import cv2
import time
import os 
import map as m


#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/get_map')
def get_map():
    resp = {}
    test = m.Map.camera_map
    for id in test:
        resp[id] = test[id]
    print(test)
    return jsonify(resp)


def gen(camera_stream, feed_type, device):

    """Video streaming generator function."""
    unique_name = (feed_type, device)

    num_frames = 0
    total_time = 0
    while True:
        time_start = time.time()
        now = time.localtime()

        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        num_frames += 1

        time_now = time.time()
        total_time += time_now - time_start
        fps = num_frames / total_time

        # write camera name
        cv2.putText(frame, cam_id, (int(20), int(20 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0], (255, 255, 255), 2)
        cv2.putText(frame, "Time : %04d/%02d/%02d %02d:%02d:%02d" %(now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec), 
        (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0], (255, 255, 255), 2)

        if feed_type == 'yolo':
            cv2.putText(frame, "FPS: %.2f" % fps, (int(20), int(60 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],
                        (255, 255, 255), 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # Remove this line for test camera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<feed_type>/<device>')
def video_feed(feed_type, device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    port_list = (5555, 5566, 5577)
    if feed_type == 'camera':
        camera_stream = import_module('camera_server').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif feed_type == 'yolo':
        camera_stream = import_module('camera_yolo').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
        return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
