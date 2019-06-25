"""
Copyright (c) 2019 Sébastien IOOSS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from absl import app, flags
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
