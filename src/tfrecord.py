import cv2
import os
import tensorflow as tf

from tqdm import tqdm
from absl import logging, flags

flags.DEFINE_integer("max_image_size", 1800, "Maximal size to rescale an image before putting in tfrecord")
flags.DEFINE_integer("images_per_tfrecord", 1000, "Number of image to write in one tfrecord file")
flags.DEFINE_integer("n_categories_to_load", -1, "Number of categories to load in place dataset. If -1 then load everything in tfrecord")

FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class TfRecordCreator:
    def __init__(self, images_per_tfrecord, n_tot_images, tf_record_dir, name):
        """

        Args:
            images_per_tfrecord: Number of images to write in one tfrecord file
            n_tot_images: Total number of images to write in tfrecord files. 0 and -1 mean no limit
            tf_record_dir: path to the dir to store the tf_record
            name: name of the tf_record. The filenames will be as "name_n.tfrecord" with n starting from 0
        """
        self.images_per_tfrecord = images_per_tfrecord

        if n_tot_images in [0, -1]:
            # Then no limit
            logging.info("load all the images")
            n_tot_images = -1
        else:
            logging.info("load {} images".format(n_tot_images))
        self.n_tot_images = n_tot_images

        self.tf_record_dir = tf_record_dir
        os.makedirs(tf_record_dir, exist_ok=True)

        self.name = name

        self.n_images_currently_loaded = 0
        self.n_images_tot_loaded = 0
        self.current_tf_record_index = 0
        self.writer = None

    def add_example_to_tfrecord(self, example):
        """
        Add an example to the tfrecord. This function take care of the creation of new files and count the number of
        loaded images
        Args:
            example: The example to save in the tfrecord

        Returns: True when all the images are loaded, else False

        """
        # Check if need to open a tf_record
        if self.n_images_currently_loaded == 0:
            tf_record_filename = os.path.join(self.tf_record_dir, self.name + '_' + str(self.current_tf_record_index) + '.tfrecord')
            logging.info('Writing {}'.format(tf_record_filename))
            self.writer = tf.io.TFRecordWriter(tf_record_filename)

        self.writer.write(example.SerializeToString())
        self.n_images_currently_loaded += 1
        self.n_images_tot_loaded += 1

        # Check if need to close a tf_record
        if self.n_images_currently_loaded == self.images_per_tfrecord:
            self.writer.close()
            self.n_images_currently_loaded = 0
            self.current_tf_record_index += 1

        return self.n_images_tot_loaded == self.n_tot_images


def process_one_image(image_path, tfrecord_writer, max_image_size):
    image = cv2.imread(image_path)
    image_shape = image.shape
    if max(image_shape) > max_image_size:
        ratio = max_image_size / max(image_shape)
        image = cv2.resize(image, (int(image_shape[1] * ratio), int(image_shape[0] * ratio)))

    img_str = cv2.imencode('.jpg', image)[1].tostring()
    feature = {'image': bytes_feature(img_str)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tfrecord_writer.add_example_to_tfrecord(example)


def prepare_place_dataset(dataset_path, tfrecords_path, tfrecord_name):
    """

    Args:
        dataset_path: ("/opt/data_large") Do not change it if you're using Docker-compose
        tfrecords_path: ("/opt/experiment/tfrecord") Do not change it if you're using Docker-compose
        tfrecord_name: the tfrecords files will be
    """
    tfrecord_writer = TfRecordCreator(FLAGS.images_per_tfrecord, -1, tfrecords_path, tfrecord_name)

    categories_names = \
        ['/a/arch', '/a/amphitheater', '/a/aqueduct', '/a/arena/rodeo', '/a/athletic_field/outdoor', '/b/badlands', '/b/balcony/exterior', '/b/bamboo_forest', '/b/barn',
         '/b/barndoor', '/b/baseball_field', '/b/beach', '/b/beach_house', '/b/beer_garden', '/b/boardwalk', '/b/boathouse', '/b/botanical_garden', '/b/bullring', '/b/butte',
         '/c/cabin/outdoor', '/c/campsite', '/c/campus', '/c/canal/natural', '/c/canal/urban', '/c/canyon', '/c/castle', '/c/church/outdoor', '/c/chalet', '/c/cliff', '/c/coast',
         '/c/corn_field', '/c/corral', '/c/cottage', '/c/courtyard', '/c/crevasse', '/d/dam', '/d/desert/vegetation', '/d/desert_road', '/d/doorway/outdoor', '/f/farm',
         '/f/field/cultivated', '/f/field/wild', '/f/field_road', '/f/fishpond', '/f/florist_shop/indoor', '/f/forest/broadleaf', '/f/forest_path', '/f/forest_road',
         '/f/formal_garden', '/g/gazebo/exterior', '/g/glacier', '/g/golf_course', '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto', '/h/hayfield', '/h/hot_spring',
         '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_floe', '/i/ice_shelf', '/i/iceberg', '/i/inn/outdoor', '/i/islet', '/j/japanese_garden', '/k/kasbah', '/k/kennel/outdoor',
         '/l/lagoon', '/l/lake/natural', '/l/lawn', '/l/library/outdoor', '/l/lighthouse', '/m/mansion', '/m/marsh', '/m/mausoleum', '/m/moat/water', '/m/mosque/outdoor',
         '/m/mountain', '/m/mountain_path', '/m/mountain_snowy', '/o/oast_house', '/o/ocean', '/o/orchard', '/p/park', '/p/pasture', '/p/pavilion', '/p/picnic_area', '/p/pier',
         '/p/pond', '/r/raft', '/r/railroad_track', '/r/rainforest', '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden', '/r/rope_bridge', '/r/ruin', '/s/schoolhouse',
         '/s/sky', '/s/snowfield', '/s/swamp', '/s/swimming_hole', '/s/synagogue/outdoor', '/t/temple/asia', '/t/topiary_garden', '/t/tree_farm', '/t/tree_house',
         '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden', '/v/viaduct', '/v/village', '/v/vineyard', '/v/volcano', '/w/waterfall',
         '/w/watering_hole', '/w/wave', '/w/wheat_field', '/z/zen_garden', '/a/alcove', '/a/artists_loft', '/b/building_facade', '/c/cemetery']
    categories_names = [x[1:] for x in categories_names]

    if FLAGS.n_categories_to_load != -1:
        categories_names = categories_names[:FLAGS.n_categories_to_load]

    dataset = []

    for category_name in tqdm(categories_names):
        if os.path.exists(os.path.join(dataset_path, category_name)):
            for file_name in os.listdir(os.path.join(dataset_path, category_name)):
                dataset.append(os.path.join(dataset_path, category_name, file_name))
        else:
            logging.warn("Categorie {} doesn't exist".format(category_name))

    logging.info("There are {} images to process".format(len(dataset)))

    for image_path in tqdm(dataset):
        if process_one_image(image_path, tfrecord_writer, FLAGS.max_image_size):
            break


def prepare_art_dataset(dataset_path, tfrecords_path, tfrecord_name):
    tfrecord_writer = TfRecordCreator(FLAGS.images_per_tfrecord, -1, tfrecords_path, tfrecord_name)

    dataset = []
    for file_name in tqdm(os.listdir(dataset_path)):
        dataset.append(os.path.join(dataset_path, file_name))
    logging.info("There are {} images to process".format(len(dataset)))
    for image_path in tqdm(dataset):
        if process_one_image(image_path, tfrecord_writer, FLAGS.max_image_size):
            break
