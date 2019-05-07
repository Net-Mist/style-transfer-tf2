import os
import tensorflow as tf
from absl import flags, logging
from tqdm import tqdm
from .model import make_models, make_transformer_model, VGG19
from .losses import discriminator_loss, generator_loss, generator_acc, discriminator_acc, cartoon_adversarial_loss, cartoon_content_loss, cartoon_generator_loss
from .datasets import Dataset
from .eval import eval_dir
from .tfrecord import prepare_art_dataset, prepare_place_dataset
from .edge_smooth import edge_smooth



FLAGS = flags.FLAGS

flags.DEFINE_enum("training_method", "adaptive_st", ["cartoon_gan", "adaptive_st", "initialization_cartoon"], "Method to train the style transfer neural network")
flags.DEFINE_string("picture_dataset_path", "/opt/picture_dataset", "Path to the directory containing the pictures for training the model")
flags.DEFINE_string("picture_tfrecord_path", "/opt/experiment/tfrecord", "Path to the directory containing the tfrecords with the training pictures")
flags.DEFINE_string("picture_tfrecord_prefix", "picture", "Prefix of the tfrecord containing the training picture")
flags.DEFINE_string("style_dataset_path", "/opt/style_dataset", "Path to the directory containing style images")
flags.DEFINE_string("style_tfrecord_path", "/opt/experiment/tfrecord", "Path to the directory containing the tfrecords with the style images")
flags.DEFINE_string("style_tfrecord_prefix", "style", "Prefix of the tfrecord containing the style images")
flags.DEFINE_string("smooth_dataset_path", "/opt/experiment/smooth_dataset", "Path to the directory containing the smoothed images. If these images don't exist then create them "
                                                                             "from style images")
flags.DEFINE_string("smooth_tfrecord_path", "/opt/experiment/tfrecord", "Path to the directory containing the tfrecords with the smoothed images. Only if training CartoonGAN")
flags.DEFINE_string("smooth_tfrecord_prefix", "smooth",
                    "Prefix of the tfrecord containing the smoothed images. Only if training CartoonGAN")  # TODO change so default is style_tfrecord_prefix + "smooth"
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate during training")
flags.DEFINE_string("pretrained_ckpt", "", "if specified then load this checkpoint before starting the training")
flags.DEFINE_integer("n_iterations", 200000, "Number of training iterations to do")
flags.DEFINE_float("discr_success_rate", 0.8, "Threshold under which we train disciminator network instead of generator part. Only useful if simult_training is false")
flags.DEFINE_bool("simult_training", False, "If True then train both generator and discriminator at the same time")
flags.DEFINE_string("training_dir", "/opt/experiment", "dir to write logs, checkpoint and validation images")
flags.DEFINE_string("test_image_dir", "/opt/experiment/test_images", "dir containing testing image for visualization during the training")


def prepare_tfrecords(dataset_path, tfrecord_path, tfrecord_prefix, art_dataset=True):
    if os.path.exists(os.path.join(tfrecord_path, tfrecord_prefix + "_0.tfrecord")):
        logging.info("tfrecords {} already exist. Skip".format(os.path.join(tfrecord_path, tfrecord_prefix)))
    else:
        logging.info("Process dataset {} to tfrecord {}".format(dataset_path, os.path.join(tfrecord_path, tfrecord_prefix)))
        if art_dataset:
            prepare_art_dataset(dataset_path, tfrecord_path, tfrecord_prefix)
        else:
            prepare_place_dataset(dataset_path, tfrecord_path, tfrecord_prefix)


def training():
    # Start by preparing the tfrecord files, or be sure tfrecord files already exist
    prepare_tfrecords(FLAGS.style_dataset_path, FLAGS.style_tfrecord_path, FLAGS.style_tfrecord_prefix)
    prepare_tfrecords(FLAGS.picture_dataset_path, FLAGS.picture_tfrecord_path, FLAGS.picture_tfrecord_prefix, art_dataset=False)

    if FLAGS.training_method == "cartoon_gan" or FLAGS.training_method == "initialization_cartoon":
        if os.path.exists(FLAGS.smooth_dataset_path):
            logging.info("smooth dataset path already exist, skip")
        else:
            logging.info("create smooth dataset")
            edge_smooth(FLAGS.style_dataset_path, FLAGS.smooth_dataset_path)
        prepare_tfrecords(FLAGS.smooth_dataset_path, FLAGS.smooth_tfrecord_path, FLAGS.smooth_tfrecord_prefix)

    # Then start the training
    t = Training()
    logging.info("training process init")
    if FLAGS.training_method == "cartoon_gan":
        t.run_simult_cartoon_gan()
    elif FLAGS.training_method == "initialization_cartoon":
        t.run_generator_initialization()
    elif FLAGS.simult_training:
        t.run_simult()
    else:
        t.run_sequential()


class Training:
    def __init__(self):
        # Prepare datasets
        self.art_dataset = Dataset(FLAGS.style_tfrecord_path, FLAGS.style_tfrecord_prefix)
        self.place_dataset = Dataset(FLAGS.picture_tfrecord_path, FLAGS.picture_tfrecord_prefix)
        if FLAGS.training_method == "cartoon_gan":
            self.smooth_dataset = Dataset(FLAGS.smooth_tfrecord_path, FLAGS.smooth_tfrecord_prefix)

        # Prepare optimizer
        self.generator_optimizer = tf.optimizers.Adam(FLAGS.learning_rate)
        self.discriminator_optimizer = tf.optimizers.Adam(FLAGS.learning_rate)

        # Prepare models
        self.encoder, self.decoder, self.discriminator = make_models()
        self.generator = tf.keras.Sequential([self.encoder, self.decoder])

        if FLAGS.training_method == "cartoon_gan" or FLAGS.training_method == "initialization_cartoon":
            self.vgg = VGG19()
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                  discriminator_optimizer=self.discriminator_optimizer,
                                                  encoder=self.encoder,
                                                  decoder=self.decoder,
                                                  discriminator=self.discriminator)
        else:
            self.transformer = make_transformer_model()
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                  discriminator_optimizer=self.discriminator_optimizer,
                                                  encoder=self.encoder,
                                                  decoder=self.decoder,
                                                  transformer=self.transformer,
                                                  discriminator=self.discriminator)

        if FLAGS.pretrained_ckpt:
            if os.path.isdir(FLAGS.pretrained_ckpt):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_ckpt)
            else:
                checkpoint_path = FLAGS.pretrained_ckpt
            logging.info("Load checkpoint {}".format(checkpoint_path))
            self.checkpoint.restore(checkpoint_path)

        self.discr_success_rate = FLAGS.discr_success_rate
        self.n_iterations = FLAGS.n_iterations

        self.test_image_dir = FLAGS.test_image_dir
        self.output_path = os.path.join(FLAGS.training_dir, "output_images")

        self.logdir = os.path.join(FLAGS.training_dir, "logs")
        self.checkpoint_prefix = os.path.join(FLAGS.training_dir, "checkpoints", "ckpt")
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.writer.set_as_default()

        self.log_rate = 100

    def save_and_eval(self, n_iter, force=False):
        if n_iter % 1000 == 0 and n_iter != 0 or force:
            logging.info("save")
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            logging.info("eval")
            eval_dir(self.encoder, self.decoder, self.test_image_dir, self.output_path, img_suffix='_{}'.format(n_iter))

    def make_logs(self, logs, global_step):
        with tf.summary.record_if(global_step % self.log_rate == 0):
            for log in logs:
                tf.summary.scalar(log, logs[log], global_step)

    def run_generator_initialization(self):
        """Method proposed in Cartoon GAN paper
        The idea is to start the training by only training the generator network with only semantic content loss
            Returns:
        """
        for global_step, picture_image in tqdm(enumerate(self.place_dataset.dataset)):
            self.save_and_eval(global_step)
            loss = self.generator_initialization_step(picture_image)
            self.make_logs({"loss": loss}, global_step)
            if global_step == self.n_iterations:
                self.save_and_eval(global_step, force=True)
                return

    def run_sequential(self):
        discr_success = 0
        alpha = 0.05
        train_generator_at_previous_step = True
        for global_step, (picture_image, art_image) in enumerate(zip(self.place_dataset.dataset, self.art_dataset.dataset)):
            self.save_and_eval(global_step)
            if discr_success >= self.discr_success_rate:
                if not train_generator_at_previous_step:
                    train_generator_at_previous_step = True
                    logging.info("Train generator")
                accuracy, loss_logs, acc_logs = self.generator_train(picture_image)
                discr_success = discr_success * (1. - alpha) + alpha * (1 - accuracy)
            else:
                if train_generator_at_previous_step:
                    train_generator_at_previous_step = False
                    logging.info("Train discriminator")
                accuracy, loss_logs, acc_logs = self.discriminator_train(picture_image, art_image)
                discr_success = discr_success * (1. - alpha) + alpha * accuracy
            self.make_logs({**loss_logs, **acc_logs, "discr_success": discr_success}, global_step)
            if global_step == self.n_iterations:
                self.save_and_eval(global_step, force=True)
                return

    def run_simult(self):
        for global_step, (picture_image, art_image) in enumerate(zip(self.place_dataset.dataset, self.art_dataset.dataset)):
            self.save_and_eval(global_step)
            self.train_both(picture_image, art_image)
            # TODO add logs

    def run_simult_cartoon_gan(self):
        logging.info("Start cartoon gan training")

        for e in self.place_dataset.dataset:
            logging.info("can access place dataset")
            break
        for e in self.art_dataset.dataset:
            logging.info("can access art dataset")
            break
        for e in self.smooth_dataset.dataset:
            logging.info("can access smooth dataset")
            break

        for global_step, (picture_image, art_image, smoothed_image) in tqdm(enumerate(zip(self.place_dataset.dataset, self.art_dataset.dataset, self.smooth_dataset.dataset))):
            self.save_and_eval(global_step)
            self.train_both_cartoon(picture_image, art_image, smoothed_image)
            if global_step == self.n_iterations:
                self.save_and_eval(global_step, force=True)
                return
            # TODO add logs

    @tf.function
    def generator_initialization_step(self, picture):
        """Initialisation phase as describe section 3.3 of the cartoon gan paper
        """
        with tf.GradientTape() as tape:
            generated = self.decoder(self.encoder(picture, training=True), training=True)
            loss = cartoon_content_loss(picture, generated, self.vgg)
        gradients = tape.gradient(loss, self.encoder.variables + self.discriminator.variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.encoder.variables + self.discriminator.variables))

        return loss

    @tf.function
    def discriminator_train(self, picture_image, art_image):
        encoded_picture = self.encoder(picture_image, training=False)
        decoded_picture = self.decoder(encoded_picture, training=False)

        with tf.GradientTape() as gen_tape:
            discriminate_encoded_picture = self.discriminator(decoded_picture, training=True)
            discriminate_picture = self.discriminator(picture_image, training=True)
            discriminate_art = self.discriminator(art_image, training=True)
            loss, loss_logs = discriminator_loss(discriminate_encoded_picture, discriminate_picture, discriminate_art)
        accuracy, acc_logs = discriminator_acc(discriminate_encoded_picture, discriminate_picture, discriminate_art)

        gradients = gen_tape.gradient(loss, self.discriminator.variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.variables))

        return accuracy, loss_logs, acc_logs

    @tf.function
    def generator_train(self, picture_image):
        with tf.GradientTape() as gen_tape:
            encoded_picture = self.encoder(picture_image, training=True)
            decoded_picture = self.decoder(encoded_picture, training=True)
            encoded_decoded_picture = self.encoder(decoded_picture, training=True)
            discriminate_encoded_picture = self.discriminator(decoded_picture, training=False)

            transformed_picture = self.transformer(picture_image, training=True)
            transformed_decoded_picture = self.transformer(decoded_picture, training=True)

            accuracy, acc_logs = generator_acc(discriminate_encoded_picture)
            loss, loss_logs = generator_loss(discriminate_encoded_picture, transformed_picture, encoded_picture, transformed_decoded_picture, encoded_decoded_picture)

        gradients = gen_tape.gradient(loss, self.encoder.variables + self.decoder.variables + self.transformer.variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.encoder.variables + self.decoder.variables + self.transformer.variables))

        return accuracy, loss_logs, acc_logs

    @tf.function
    def train_both(self, picture_image, art_image):
        with tf.GradientTape() as gen_tape:
            encoded_picture = self.encoder(picture_image, training=True)
            decoded_picture = self.decoder(encoded_picture, training=True)
            encoded_decoded_picture = self.encoder(decoded_picture, training=True)
            transformed_decoded_picture = self.transformer(decoded_picture, training=True)
            transformed_picture = self.transformer(picture_image, training=True)
            regularization_loss_gen = tf.reduce_sum(self.encoder.losses + self.decoder.losses)

            with tf.GradientTape() as disc_tape:
                discriminate_decoded_picture = self.discriminator(decoded_picture, training=True)
                gen_loss = generator_loss(discriminate_decoded_picture, transformed_picture, encoded_picture, transformed_decoded_picture,
                                          encoded_decoded_picture) + regularization_loss_gen

                discriminate_picture = self.discriminator(picture_image, training=True)
                discriminate_art = self.discriminator(art_image, training=True)
                regularization_loss_disc = tf.reduce_sum(self.discriminator.losses)
                disc_loss = discriminator_loss(discriminate_decoded_picture, discriminate_picture, discriminate_art) + regularization_loss_disc

        generator_acc(discriminate_decoded_picture)
        discriminator_acc(discriminate_decoded_picture, discriminate_picture, discriminate_art)

        tf.summary.scalar("generator_loss/5_reg", regularization_loss_gen)
        tf.summary.scalar("discriminator_loss/5_reg", regularization_loss_disc)

        gen_gradients = gen_tape.gradient(gen_loss, self.encoder.variables + self.decoder.variables + self.transformer.variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.encoder.variables + self.decoder.variables + self.transformer.variables))

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.variables))

    @tf.function
    def train_both_cartoon(self, picture_image, art_image, smoothed_image):
        with tf.GradientTape() as gen_tape:
            generated_image = self.generator(picture_image)
            content_loss = cartoon_content_loss(picture_image, generated_image, self.vgg)
            gen_loss = cartoon_generator_loss(generated_image)
            gen_loss += content_loss
        with tf.GradientTape() as disc_tape:
            disc_generated_image = self.discriminator(generated_image)
            disc_art_image = self.discriminator(art_image)
            disc_smoothed_image = self.discriminator(smoothed_image)
            disc_loss = cartoon_adversarial_loss(disc_art_image, disc_smoothed_image, disc_generated_image)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.variables))

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.variables))
