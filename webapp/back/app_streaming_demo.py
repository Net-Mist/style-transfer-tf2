import threading
import cv2
from flask import Flask, render_template, Response
import time


class Camera(threading.Thread):
    def __init__(self):
        super().__init__()
        self.camera_port = 0
        self.fps = 30
        self.video_capture = None

        self.running = False

        self.lock = threading.Lock()
        self.frame = None

        self.last_get_frame_time = 0
        self.video_capture = cv2.VideoCapture(self.camera_port)

    def run(self) -> None:
        self.running = True
        previous_time = time.time()
        while self.running:
            current_time = time.time()
            if current_time < previous_time + 1 / self.fps:
                time.sleep(previous_time + 1 / self.fps - current_time)
            success, image = self.video_capture.read()
            try:
                ret, jpeg_encoded = cv2.imencode('.jpg', image)
            except Exception as e:
                print(e)
                continue
            with self.lock:
                self.frame = jpeg_encoded.tobytes()
            previous_time = current_time

        self.video_capture.release()

    def get_frame(self):
        current_time = time.time()

        if current_time - self.last_get_frame_time < 1 / self.fps:
            time.sleep(self.last_get_frame_time + 1 / self.fps - current_time)
        self.last_get_frame_time = current_time
        with self.lock:
            return self.frame


app = Flask(__name__)

camera = None


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    global camera
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    global camera
    camera = Camera()
    camera.start()
    app.run(host='0.0.0.0')


if __name__ == '__main__':
    main()
