import os
import cv2
import numpy as np
from absl import flags, logging

flags.DEFINE_integer("blur_size", 25, "Size of the kernel of the cv2.blur operation")
flags.DEFINE_bool("blur_all_image", True, "If true then blur the whole image, else only blur the edges")

FLAGS = flags.FLAGS


def edge_smooth(input_dir, output_dir):
    """
    Load all the images in input_dir, blur them as describe in the paper "CartoonGAN" and write the results in output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    files.sort()
    logging.info("Process {} files".format(len(files)))

    dilate_kernel = np.ones((5, 5), np.uint8)
    # Process the images as describe in the paper "CartoonGAN"
    for file in files:
        image = cv2.cvtColor(cv2.imread(os.path.join(input_dir, file)), cv2.COLOR_BGR2RGB)

        if not FLAGS.blur_all_image:
            # 1 Detect edge pixel
            logging.debug("Apply Canny edge detector")
            edges = cv2.Canny(image, 100, 200)
            logging.debug("{}, {}".format(edges.shape, edges.max()))

            # 2 Dilate edge regions
            logging.debug("Dilate")
            dilation = cv2.dilate(edges, dilate_kernel, iterations=3)
            dilation = np.stack([dilation] * 3, axis=2)
            dilation_bool = dilation / 255

        # 3 Apply gaussian smoothing
        logging.debug("Blur")
        blured = cv2.blur(image, (FLAGS.blur_size, FLAGS.blur_size))

        if FLAGS.blur_all_image:
            final_image = blured
        else:
            final_image = (image * (1 - dilation_bool) + blured * dilation_bool).astype(np.uint8)
        logging.debug("{}".format(final_image.dtype))

        # and save
        cv2.imwrite(os.path.join(output_dir, file), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
