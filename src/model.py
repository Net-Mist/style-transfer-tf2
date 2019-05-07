from absl import logging, flags
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import tf_utils

# Define model flags
FLAGS = flags.FLAGS

flags.DEFINE_enum("increase_size_layer_type", "default", ["default", "unpool", "deconv"],
                  "Type of layer to use in the decoder part. Default use the type specified in research paper (unpool for adaptive st, deconv for cartoon GAN)")
flags.DEFINE_enum("norm_layer", "default", ["default", "instance_norm", "batch_norm"],
                  "Type of layer to use for normalization. Default use the type specified in research paper (instance_norm for adaptive st, batch_norm for cartoon GAN)")
flags.DEFINE_bool("mobilenet", False, "Build model with mobilenet optimization (depthwise convolution...)")
flags.DEFINE_enum("model", "default", ["default", "adaptive_st", "cartoon_gan"],
                  "Model topology to use. If default then use the topology corresponding to training_method")

flags.DEFINE_integer("n_filter_generator", 32, "Number of filters in first conv layer of generator (encoder-decoder)")
flags.DEFINE_integer("n_filter_discriminator", 64, "Number of filters in first conv layer of discriminator")
flags.DEFINE_float("l2_reg", 0.001, "l2 regularization weigh to apply")
flags.DEFINE_integer("transformer_kernel_size", 10, "Size of kernel we apply to the input_tensor in the transformer model if using adaptive_st training method")


#################
# Custom layers #
#################
class UnpoolLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3]

    def get_config(self):
        base_config = super(UnpoolLayer, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return tf.image.resize(inputs, tf.shape(inputs)[1:3] * 2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


class InstanceNormLayer(keras.layers.Layer):
    """
    See:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization.py

    """

    def __init__(self, **kwargs):
        self.epsilon = 1e-5
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([shape[0], shape[1], shape[2], shape[3]])

    def build(self, input_shape):
        depth = (input_shape[3],)
        self.scale = self.add_weight(shape=depth,
                                     name='gamma',
                                     initializer=keras.initializers.get('ones'))
        self.offset = self.add_weight(shape=depth,
                                      name='gamma',
                                      initializer=keras.initializers.get('zeros'))

        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset


class ReflectPadLayer(keras.layers.Layer):
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([shape[0], shape[1] + self.pad_size * 2, shape[2] + self.pad_size * 2, shape[3]])

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], "SYMMETRIC")


class CenterLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return inputs * 2. - 1.


##############
# Custom ops #
##############
def relu6(x):
    return tf.keras.activations.relu(x, max_value=6)


def inverted_res_block(inputs, stride, in_channels, out_channels, norm_layer, expansion=1):
    x = inputs
    x = tf.keras.layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    # Depthwise
    if stride == 2:
        x = norm_layer()(x)
    x = ReflectPadLayer(1)(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='valid' if stride == 1 else 'valid')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    # Project
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = norm_layer()(x)

    if in_channels == out_channels and stride == 1:
        return keras.layers.Add()([inputs, x])
    return x


def make_models():
    """build all the models based on the arguments provided via absl

    Returns: encoder_model, decoder_model, discriminator_model

    """
    if FLAGS.model == "default" and FLAGS.training_method == "adaptive_st" or FLAGS.model == "adaptive_st":
        logging.info("define adaptive_st model")
        if FLAGS.norm_layer == "instance_norm" or FLAGS.norm_layer == "default":
            norm_layer = InstanceNormLayer
        else:
            norm_layer = tf.keras.layers.BatchNormalization
            logging.warning("Use unusual norm layer for this model")
        if FLAGS.increase_size_layer_type == "default" or FLAGS.increase_size_layer_type == "unpool":
            increase_size_layer = UnpoolLayer
        else:
            increase_size_layer = tf.keras.layers.Conv2DTranspose
            raise Exception("Not yet implemented")

        discriminator_model = make_discriminator_model_adaptive_style_transfer(norm_layer)

        if not FLAGS.mobilenet:
            encoder_model = make_encoder_model_adaptive_style_transfer(norm_layer)
            decoder_model = make_decoder_model_adaptive_style_transfer(encoder_model.output_shape[1:], norm_layer)
        else:
            logging.info("Use mobilenet version")
            encoder_model = make_encoder_model_mobilenet(norm_layer)
            decoder_model = make_decoder_model_mobilenet(encoder_model.output_shape[1:], norm_layer)

    else:
        logging.info("define cartoon_gan model")
        if FLAGS.norm_layer == "batch_norm" or FLAGS.norm_layer == "default":
            norm_layer = tf.keras.layers.BatchNormalization
        else:
            norm_layer = InstanceNormLayer
            logging.warning("Use unusual norm layer for this model")
        if FLAGS.increase_size_layer_type == "default" or FLAGS.increase_size_layer_type == "deconv":
            increase_size_layer = tf.keras.layers.Conv2DTranspose
        else:
            increase_size_layer = UnpoolLayer
            raise Exception("Not yet implemented")

        encoder_model = make_encoder_model_cartoon(norm_layer)
        decoder_model = make_decoder_model_cartoon(encoder_model.output_shape[1:], norm_layer)
        discriminator_model = make_discriminator_model_cartoon(norm_layer)

    return encoder_model, decoder_model, discriminator_model


def make_encoder_model_adaptive_style_transfer(norm_layer):
    """encoder model following https://arxiv.org/pdf/1807.10201.pdf
    Returns: encoder model
    """
    model = keras.Sequential(name="Encoder")
    model.add(norm_layer(input_shape=(FLAGS.image_size, FLAGS.image_size, 3), dtype=tf.float32))
    model.add(ReflectPadLayer(15))

    def add_conv(n_filter, strides):
        model.add(keras.layers.Conv2D(n_filter, 3, strides, 'VALID', kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg)))
        model.add(norm_layer())
        model.add(keras.layers.Activation("relu"))

    add_conv(FLAGS.n_filter_generator, 1)
    add_conv(FLAGS.n_filter_generator, 2)
    add_conv(FLAGS.n_filter_generator * 2, 2)
    add_conv(FLAGS.n_filter_generator * 4, 2)
    add_conv(FLAGS.n_filter_generator * 8, 2)

    return model


def make_decoder_model_adaptive_style_transfer(input_shape, norm_layer):
    """decoder model following https://arxiv.org/pdf/1807.10201.pdf
    Returns: decoder model
    """

    x = keras.layers.Input(shape=input_shape, dtype=tf.float32)
    inputs = x

    def residual_block(x, dim, kernel_size=3, s=1):
        pad = int((kernel_size - 1) / 2)
        y = ReflectPadLayer(pad)(x)
        y = keras.layers.Activation("relu")(norm_layer()(keras.layers.Conv2D(dim, kernel_size, s, 'VALID', kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(y)))
        y = ReflectPadLayer(pad)(y)
        y = norm_layer()(keras.layers.Conv2D(dim, kernel_size, s, 'VALID', kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(y))
        return keras.layers.Add()([x, y])

    # Now stack 9 residual blocks
    num_kernels = FLAGS.n_filter_generator * 8
    for i in range(9):
        x = residual_block(x, num_kernels)

    # Decode image.
    for i in range(4):
        x = UnpoolLayer()(x)
        x = keras.layers.Conv2D(FLAGS.n_filter_generator * 2 ** (3 - i), 3, 1, "SAME", kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(x)
        x = keras.layers.Activation("relu")(norm_layer()(x))

    x = ReflectPadLayer(3)(x)

    x = keras.layers.Activation("sigmoid")(keras.layers.Conv2D(3, 7, 1, "VALID", kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(x))
    x = CenterLayer()(x)

    model = keras.Model(inputs=inputs, outputs=x, name="Decoder")

    return model


def make_encoder_model_mobilenet(norm_layer):
    x = keras.layers.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3), dtype=tf.float32)
    inputs = x
    x = norm_layer()(x)
    x = ReflectPadLayer(15)(x)

    def add_conv(n_filter_new, strides, x):
        x = keras.layers.DepthwiseConv2D(3, strides=strides, use_bias=False)(x)
        x = InstanceNormLayer()(x)
        x = tf.keras.layers.Activation(relu6)(x)
        x = keras.layers.Conv2D(n_filter_new, 1, 1, activation=relu6, use_bias=False, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(x)
        x = InstanceNormLayer()(x)
        x = tf.keras.layers.Activation(relu6)(x)
        return x

    # First conv is a normal conv
    x = keras.layers.Conv2D(FLAGS.n_filter_generator, 3, 1, 'VALID', kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(x)
    x = norm_layer()(x)
    x = keras.layers.Activation(relu6)(x)
    # x = add_conv(n_filter, 1, x)

    # Then use DWConv
    x = add_conv(FLAGS.n_filter_generator, 2, x)
    x = add_conv(FLAGS.n_filter_generator * 2, 2, x)
    x = add_conv(FLAGS.n_filter_generator * 4, 2, x)
    x = add_conv(FLAGS.n_filter_generator * 8, 2, x)

    model = keras.Model(inputs=inputs, outputs=x, name="Encoder")
    return model


def make_decoder_model_mobilenet(input_shape, norm_layer):
    x = keras.layers.Input(shape=input_shape, dtype=tf.float32)
    inputs = x

    # Residual part
    num_kernels = FLAGS.n_filter_generator * 8
    kernel_size = 3
    pad = int((kernel_size - 1) / 2)
    for i in range(9):
        x = inverted_res_block(x, 1, num_kernels, num_kernels, norm_layer)
        x = inverted_res_block(x, 1, num_kernels, num_kernels, norm_layer)

    # Decode image
    for i in range(4):
        x = UnpoolLayer()(x)
        x = inverted_res_block(x, 1, FLAGS.n_filter_generator * 2 ** (3 - i + 1), FLAGS.n_filter_generator * 2 ** (3 - i), norm_layer)

    x = ReflectPadLayer(3)(x)

    x = keras.layers.Activation("sigmoid")(keras.layers.Conv2D(3, 7, 1, "VALID", kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(x))
    x = CenterLayer()(x)

    model = keras.Model(inputs=inputs, outputs=x, name="Decoder")
    return model


def make_discriminator_model_adaptive_style_transfer(norm_layer):
    """
    Discriminator agent, that provides us with information about image plausibility at different scales.
    Returns:
        Image estimates at different scales.
    """

    image = keras.layers.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3), dtype=tf.float32)
    h0 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 2, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(image)))
    h0_pred = keras.layers.Conv2D(1, 5)(h0)

    h1 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 2, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(h0)))
    h1_pred = keras.layers.Conv2D(1, 10)(h1)

    h2 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 4, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(h1)))

    h3 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 8, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(h2)))
    h3_pred = keras.layers.Conv2D(1, 10)(h3)

    h4 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 8, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(h3)))

    h5 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 16, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(h4)))
    h5_pred = keras.layers.Conv2D(1, 6)(h5)

    h6 = keras.layers.LeakyReLU()(norm_layer()(keras.layers.Conv2D(FLAGS.n_filter_discriminator * 16, 5, 2, kernel_regularizer=keras.regularizers.l2(FLAGS.l2_reg))(h5)))
    h6_pred = keras.layers.Conv2D(1, 3)(h6)

    model = keras.Model(inputs=image, outputs=[h0_pred, h1_pred, h3_pred, h5_pred, h6_pred], name="Discriminator")

    return model


def make_transformer_model():
    """
    This is a simplified version of transformer block described in the paper
    https://arxiv.org/abs/1807.10201.
    Returns:
        Transformed tensor
    """
    model = keras.Sequential(name="Transformer")
    model.add(keras.layers.AvgPool2D(FLAGS.transformer_kernel_size, strides=1, padding="same"))
    return model


def make_encoder_model_cartoon(norm_layer):
    """
        Follow the description in the paper http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf
        """
    x = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3))
    x_input = x

    # flat convolution stage
    x = tf.keras.layers.Conv2D(64, 7, padding="same")(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Down convolution stage
    for n_filters in [128, 256]:
        x = tf.keras.layers.Conv2D(n_filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.Conv2D(n_filters, 3, 1, padding="same")(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    model = tf.keras.Model(inputs=[x_input], outputs=[x], name="Encoder")
    return model


def make_decoder_model_cartoon(input_shape, norm_layer):
    x = tf.keras.Input(shape=input_shape)
    x_input = x

    # Residual part
    for _ in range(8):
        x_residual = x
        x = tf.keras.layers.Conv2D(256, 3, 1, padding="same")(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(256, 3, 1, padding="same")(x)
        x = norm_layer()(x)
        x = tf.keras.layers.Add()([x, x_residual])

    # Up-convolution
    for n_filters in [128, 64]:
        x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.Conv2D(n_filters, 3, 1, padding="same")(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(3, 7, padding="same")(x)

    model = tf.keras.Model(inputs=[x_input], outputs=[x], name="Decoder")
    return model


def make_discriminator_model_cartoon(norm_layer):
    x = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3))
    x_input = x

    # Flat convolution
    x = tf.keras.layers.Conv2D(32, 3, 1, "same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Down convolution stage
    for n_filters in [64, 128]:
        x = tf.keras.layers.Conv2D(n_filters, 3, 2, "same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(n_filters * 2, 3, 1, "same")(x)
        x = norm_layer()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 3, 1, "same")(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(1, 3, 1, "same")(x)

    model = tf.keras.Model(inputs=[x_input], outputs=[x], name="generator")
    return model


def VGG19():
    layers = tf.keras.layers

    img_input = layers.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3))
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    save_x = x

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    model = tf.keras.models.Model(img_input, x, name='vgg16')

    # Load weights.
    weights_path_no_top = ('https://github.com/fchollet/deep-learning-models/'
                           'releases/download/v0.1/'
                           'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    weights_path = tf.keras.utils.get_file(
        'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        weights_path_no_top,
        cache_subdir='models',
        file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path)

    sub_model = tf.keras.models.Model(img_input, save_x, name='VGG')

    return sub_model
