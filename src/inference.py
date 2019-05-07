import cv2
import os
import logging
import time
import numpy as np
import tensorflow as tf
import tqdm
from absl import flags

flags.DEFINE_string("inference_model_path", "", "Path of the model to load for inference. Can be a saved model or a tflite model")
flags.DEFINE_string("inference_input_dir", "", "Path of the directory containing images to transform")
flags.DEFINE_string("inference_output_dir", "", "Path of the directory containing transformed images")
flags.DEFINE_bool("inference_tflite", False, "If True then use tflite to do inference instead of saved model")

FLAGS = flags.FLAGS


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
    def __init__(self):
        """
        Load the tensorflow saved model that should be in path FLAGS.inference_model_path
        """
        self.loaded = tf.saved_model.load(FLAGS.inference_model_path)
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


class TfLiteInference:
    def __init__(self):
        """
        Load the tflite model that should be in path FLAGS.inference_model_path
        """
        self.interpreter = tf.lite.Interpreter(model_path=FLAGS.inference_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logging.info(self.input_details[0])
        logging.info(self.output_details)
        logging.info("TfLiteInference init done")

    def infer(self, image):
        image = image_preprocess(image, (640, 480))
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        tflite_results = self.interpreter.get_tensor(self.output_details[0]['index'])
        return image_postprocess(tflite_results)


def load_model_and_infer_dir():
    input_dir = FLAGS.inference_input_dir
    output_dir = FLAGS.inference_output_dir
    if FLAGS.inference_tflite:
        engine = TfLiteInference()
        video_frame_size = (640, 480)
    else:
        engine = SavedModelInference()
        video_frame_size = (1920, 1080)

    for file_name in os.listdir(input_dir):
        _, extension = os.path.splitext(file_name)
        if extension == ".jpg":
            logging.info("Process image {}".format(file_name))
            image = cv2.imread(os.path.join(input_dir, file_name))
            output_image = engine.infer(image)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, output_image)
        if extension == ".mp4":
            logging.info("Process video {}".format(file_name))
            start_time = time.time()
            cap = cv2.VideoCapture(os.path.join(input_dir, file_name))
            max_frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = cv2.VideoWriter(os.path.join(output_dir, file_name), cv2.VideoWriter_fourcc(*'MP4V'), 30, video_frame_size)
            for _ in tqdm.tqdm(range(max_frame_nb)):
                ret, frame = cap.read()
                out_image = engine.infer(frame)
                out.write(out_image)
            stop_time = time.time()
            out.release()
            cap.release()
            logging.info("Process {:.3f} frames in {:.3f} seconds : {:.3f} fps".format(max_frame_nb, stop_time - start_time, max_frame_nb / (stop_time - start_time)))
