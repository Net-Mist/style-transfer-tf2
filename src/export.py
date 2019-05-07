import os
import tensorflow as tf
from absl import flags, logging
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from .model import make_models

flags.DEFINE_string("ckpt_to_export", "/opt/saved/", "If is a dir, then load the last ckpt inside and export it. If is a file then export it")
flags.DEFINE_string("export_path", "/opt/export", "Path to write the exported model")
flags.DEFINE_enum("trt_precision_mode", "FP32", ["FP16", "FP32"], "Precision mode for tensorRT model")  # TODO int8 is a bit more complexe, will come later
flags.DEFINE_enum("export_format", "saved_model", ["saved_model", "tensorrt", "tflite"], "In which format the tensorflow model need to be exported")

FLAGS = flags.FLAGS


def load_model():
    """
    Returns: the tf.keras model loaded. All the parameter of this model are variables managed by absl.flags, as for the training
    """

    if os.path.isdir(FLAGS.ckpt_to_export):
        logging.info("{} is a dir".format(FLAGS.ckpt_to_export))
        # Find the good file to load
        files = [f for f in os.listdir(FLAGS.ckpt_to_export) if ".index" in f]
        # Here files is a list "ckpt-XXXX.index"
        save_nums = [int(f[5:-6]) for f in files]
        save_nums.sort()
        biggest_num = save_nums[-1]
        ckpt_full_path = os.path.join(FLAGS.ckpt_to_export, "ckpt-{}".format(biggest_num))
    else:
        ckpt_full_path = FLAGS.ckpt_to_export

    encoder, decoder, _ = make_models()

    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(ckpt_full_path)
    logging.info("Load {}".format(ckpt_full_path))
    return tf.keras.Sequential([encoder, decoder])


def export():
    """Load the model and export it in one of the format specified by FLAGS.export_format
    IMPORTANT : the export with TensorRT is still work in progress inside Tensorflow. It doesn't work with the 2.0a official version but does
    with a version compiled from master. It will probably evolve in a near futur
    """
    model = load_model()
    logging.info("Model loaded. Summary:")
    model.summary()
    logging.info("Last ops: {}".format([node.op.name for node in model.outputs]))

    if FLAGS.export_format == "saved_model" or FLAGS.export_format == "tensorrt":
        saved_model_path = os.path.join(FLAGS.export_path, "saved_model")
        tf.saved_model.save(model, saved_model_path)
        if FLAGS.export_format == "tensorrt":
            # Convert the SavedModel using TF-TRT, see https://github.com/aaroey/tensorflow/blob/tftrt20/tftrt20/test.py
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(use_function_backup=False, precision_mode=FLAGS.trt_precision_mode)
            converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path, conversion_params=conversion_params)
            converter.convert()
            converter.save(os.path.join(FLAGS.export_path, "trt"))
    elif FLAGS.export_format == "tflite":
        concrete_func = tf.function(lambda x: model(x, training=False)).get_concrete_function(x=tf.TensorSpec((1, 480, 640, 3), tf.float32))
        converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
        tflite_model = converter.convert()
        open(FLAGS.export_path, "wb").write(tflite_model)
