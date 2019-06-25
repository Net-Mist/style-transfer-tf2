import time
import threading
import cv2
import os
import tensorflow as tf
import numpy as np
import logging
from flask import Flask, url_for, redirect, Response, jsonify
from flask_cors import CORS

# Global var before starting the API
camera = None
inference_engine = None
app = Flask(__name__)
CORS(app)

# Usefull methods and classes


class Camera(threading.Thread):
    def __init__(self):
        super().__init__()
        self.camera_port = 0
        self.fps = 30
        self.video_capture = None

        self.running = False

        self.lock = threading.Lock()
        self.frame = None  # jpeg encoded to bytes
        self.numpy_frame = None

        self.video_capture = cv2.VideoCapture(self.camera_port)

    def run(self) -> None:
        self.running = True
        previous_time = time.time()
        while self.running:
            current_time = time.time()
            if current_time < previous_time + 1 / self.fps:
                time.sleep(previous_time + 1 / self.fps - current_time)
            success, self.numpy_frame = self.video_capture.read()
            try:
                ret, jpeg_encoded = cv2.imencode('.jpg', self.numpy_frame)
            except Exception as e:
                print(e)
                continue
            with self.lock:
                self.frame = jpeg_encoded.tobytes()
            previous_time = current_time

        self.video_capture.release()

    def get_frame(self):
        with self.lock:
            return self.frame


def image_preprocess(image, resized_shape):
    """
    Preprocess the image before feeding the neural network:
      - convert from BGR (opencv) to RGB
      - resize
      - center the pixel values between -1 and 1
      - add a new dimension for the batch
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image = image / 127.5 - 1.
    image = image[np.newaxis, :, :, :].astype(np.float32)
    return image


def image_postprocess(image):
    """
    postprocess the image after the neural network part
      - remove the image from the batch
      - set pixel values between 0 and 255
      - convert from RGB to BGR for opencv
    """
    image = image[0, :, :, :]
    image = ((image + 1) * 127.5).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class SavedModelInference:
    def __init__(self, inference_model_path):
        """
        Load the tensorflow saved model
        """
        self.loaded = tf.saved_model.load(inference_model_path)
        logging.info(list(self.loaded.signatures.keys()))
        self.interpreter = self.loaded.signatures["serving_default"]
        logging.info(self.interpreter)
        logging.info(self.interpreter.inputs)
        logging.info(self.interpreter.structured_outputs)
        logging.info(list(self.interpreter.structured_outputs.keys())[0])
        self.output_label = list(self.interpreter.structured_outputs.keys())[0]

    def infer(self, image):
        image = image_preprocess(image, (1920, 1080))
        decoded_image = self.interpreter(tf.constant(image))[self.output_label].numpy()
        return image_postprocess(decoded_image)[4:-4, :, :]


def inference(image):
    image = inference_engine.infer(image)
    ret, jpeg_encoded = cv2.imencode('.jpg', image)
    frame = jpeg_encoded.tobytes()
    return frame


@app.route("/")
def index():
    return redirect(url_for('static', filename='index.html'))


@app.route("/video_feed")
def video_feed():
    def gen_image():
        last_get_frame_time = 0
        while True:
            current_time = time.time()
            if current_time - last_get_frame_time < 1 / camera.fps:
                time.sleep(last_get_frame_time + 1 / camera.fps - current_time)
            last_get_frame_time = current_time

            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/inference_feed")
def inference_feed():
    def gen_inference():
        last_get_frame_time = 0
        while True:
            current_time = time.time()
            if current_time - last_get_frame_time < 1 / camera.fps:
                time.sleep(last_get_frame_time + 1 / camera.fps - current_time)
            last_get_frame_time = current_time

            numpy_frame = camera.numpy_frame

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + inference(numpy_frame) + b'\r\n')

    return Response(gen_inference(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_configure():
    """Return the possible cameras port

    Returns: list of int for the ports

    """
    possibles_id = [int(file[5:]) for file in os.listdir("/dev") if "video" in file]
    return {"camera_ports": possibles_id}, 200


def post_configure(image_size, camera_port, model_name):
    pass


def get_raw_image():
    pass


def get_processed_image():
    pass


def main():
    global camera, inference_engine
    camera = Camera()
    camera.start()

    print("load TF")
    inference_engine = SavedModelInference("/opt/model")
    print("TF loaded")
    app.run(host='0.0.0.0', threaded=True, port=8080)


if __name__ == '__main__':
    main()
