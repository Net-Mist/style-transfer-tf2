import tensorflow as tf
from absl.testing import absltest
from absl import flags

from src import datasets
from src import model

FLAGS = flags.FLAGS


class ModelTests(absltest.TestCase):
    def test_make_encoder_model_adaptive_style_transfer(self):
        flags.FLAGS.set_default("model", "adaptive_st")
        flags.FLAGS.set_default("image_size", 768)
        m = model.make_encoder_model_adaptive_style_transfer(model.InstanceNormLayer)
        self.assertEqual(m.output_shape, (None, 48, 48, 256))
        self.assertEqual(m.count_params(), 398694)

    def test_make_decoder_model_adaptive_style_transfer(self):
        m = model.make_decoder_model_adaptive_style_transfer((48, 48, 256), model.InstanceNormLayer)
        o = m(tf.zeros((1, 48, 48, 256)))
        self.assertEqual(m.count_params(), 11613699)
        self.assertEqual(o.shape, (1, 768, 768, 3))

    def test_make_encoder_model_cartoon(self):
        m = model.make_encoder_model_cartoon(model.InstanceNormLayer)
        self.assertEqual(m.output_shape, (None, 192, 192, 256))
        self.assertEqual(m.count_params(), 1117056)

    def test_make_decoder_model_cartoon(self):
        m = model.make_decoder_model_cartoon((192, 192, 256), model.InstanceNormLayer)
        self.assertEqual(m.output_shape, (None, 768, 768, 3))
        self.assertEqual(m.count_params(), 10012611)

    def test_VGG19(self):
        m = model.VGG19()
        self.assertEqual(m.output_shape, (None, 96, 96, 512))
        self.assertEqual(m.count_params(), 10585152)

