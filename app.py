from absl import app, flags, logging
from src import datasets, training, export, inference

flags.declare_key_flag("image_size")
flags.declare_key_flag("batch_size")

flags.adopt_module_key_flags(training)
flags.adopt_module_key_flags(export)
flags.adopt_module_key_flags(inference)

flags.DEFINE_enum("action", None, ["training", "export", "infer", "train_and_export"], "What the program need to do")
flags.mark_flag_as_required("action")
FLAGS = flags.FLAGS


def main(argv):
    del argv
    if "training" in FLAGS.action:
        training.training()
    if "export" in FLAGS.action:
        export.export()
    if "infer" in FLAGS.action:
        inference.load_model_and_infer_dir()


if __name__ == '__main__':
    app.run(main)
