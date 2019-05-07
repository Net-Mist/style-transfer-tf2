import os
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

# Key flags
flags.DEFINE_integer("image_size", 768, "size of the image for training the neural network", lower_bound=0)
flags.DEFINE_integer("batch_size", 1, "number of images in a batch", lower_bound=1)

# Other flags
flags.DEFINE_float("hue_shift", 0.05, "hue shift for data augmentation", lower_bound=0)
flags.DEFINE_float("saturation_shift", 0.05, "saturation shift for data augmentation", lower_bound=0)
flags.DEFINE_float("saturation_scale", 0.05, "saturation scale for data augmentation, Scale is apply before shift", lower_bound=0)
flags.DEFINE_float("value_shift", 0.05, "value shift for data augmentation", lower_bound=0)
flags.DEFINE_float("value_scale", 0.05, "value scale for data augmentation, Scale is apply before shift", lower_bound=0)
flags.DEFINE_integer("buffer_size", 50 * 1024 ** 2, "size in bytes of the data reader buffer")


# TODO add dataset creation without tfrecord

class Dataset:
    def __init__(self, tf_record_dir, tf_record_prefix):
        """return a tf dataset to provide data for the training

        Args:
            tf_record_dir: Path of the dir where the tfrecord are stored
            tf_record_prefix: prefix of the tfrecord files to load
        """
        self.tfrecord_names = [os.path.join(tf_record_dir, file) for file in os.listdir(tf_record_dir) if file.startswith(tf_record_prefix)]

        self.image_size = FLAGS.image_size

        self.hue_shift = FLAGS.hue_shift
        self.saturation_shift = FLAGS.saturation_shift
        self.saturation_scale = FLAGS.saturation_scale
        self.value_shift = FLAGS.value_shift
        self.value_scale = FLAGS.value_scale

        self.dataset = tf.data.TFRecordDataset(self.tfrecord_names, buffer_size=FLAGS.buffer_size, num_parallel_reads=2, compression_type=None). \
            repeat(-1). \
            shuffle(10). \
            map(self.map_fn, num_parallel_calls=2). \
            batch(FLAGS.batch_size). \
            prefetch(None)

    def map_fn(self, example_proto):
        """Preprocessing and data augmentation
        Start by random scaling and cropping the image, then transform to HSV and change a little each value

        Returns: a tf.Tensor of type float32 and values between -1 and 1
        """
        features = {"image": tf.io.FixedLenFeature((), tf.string)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image"], channels=3)

        # Random scale
        scale_ratio = tf.random.uniform([1], 0.8, 1.2) * tf.cast(self.image_size / tf.math.reduce_min(tf.shape(image)[:2]), tf.float32)
        size = tf.cast(scale_ratio * tf.cast(tf.shape(image)[:2], tf.float32), tf.int32)
        image = tf.image.resize(image, size)

        # Resize
        pad_h = tf.cast(tf.math.maximum((self.image_size - tf.shape(image)[0]) / 2 + 1, 0), tf.int32)
        pad_w = tf.cast(tf.math.maximum((self.image_size - tf.shape(image)[1]) / 2 + 1, 0), tf.int32)
        image = tf.pad(image, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="SYMMETRIC")
        image = tf.image.random_crop(image, (self.image_size, self.image_size, 3))

        # HSV transform
        image = tf.image.rgb_to_hsv(image)  # pixel value are in 1, 1, 255
        image_h = image[:, :, 0]
        image_s = image[:, :, 1]
        image_v = image[:, :, 2]

        image_h += tf.random.uniform([1], -self.hue_shift, self.hue_shift)
        image_s = image_s * tf.random.uniform([1], 1. / (1. + self.saturation_scale), 1. + self.saturation_scale) + tf.random.uniform([1], -self.saturation_shift,
                                                                                                                                      self.saturation_shift)
        image_v = (image_v / 255 * tf.random.uniform([1], 1. / (1. + self.value_scale), 1. + self.value_scale) + tf.random.uniform([1], -self.value_shift, self.value_shift)) * 255

        image_h = tf.clip_by_value(image_h, 0, 1)
        image_s = tf.clip_by_value(image_s, 0, 1)
        image_v = tf.clip_by_value(image_v, 0, 255)

        image = tf.stack([image_h, image_s, image_v], axis=2)
        image = tf.image.hsv_to_rgb(image)

        image = image / 127.5 - 1.
        image = tf.cast(image, tf.float32)
        return image
