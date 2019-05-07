import tensorflow as tf
import os
import cv2
import numpy as np
import logging

# from .model import make_encoder_model, make_decoder_model
# from .generic.src.imgprovider.images_generator import ImageGeneratorCamera

# TODO clean


def eval(encoder, decoder, image):
    image = image / 127.5 - 1.
    image = image[np.newaxis, :, :, :].astype(np.float32)
    encoded_image = encoder(image, training=False)
    decoded_image = decoder(encoded_image, training=False).numpy()  # between -1 and 1
    output_image = ((decoded_image[0, :, :, :] + 1) * 127.5).astype(np.uint8)

    return output_image


def eval_dir(encoder, decoder, images_dir, output_path, img_suffix=""):
    os.makedirs(output_path, exist_ok=True)
    for file in os.listdir(images_dir):
        image = cv2.cvtColor(cv2.imread(os.path.join(images_dir, file)), cv2.COLOR_BGR2RGB)
        output_image = eval(encoder, decoder, image)
        cv2.imwrite(os.path.join(output_path, "{}{}.jpg".format(file.split('.')[0], img_suffix)), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


def eval_cam(encoder, decoder, video_port):
    # gen = ImageGeneratorCamera(video_port, (1920, 1080))
    # gen = ImageGeneratorCamera(video_port, (1280, 720))
    gen = ImageGeneratorCamera(video_port, (1280, 720))

    k = 0
    while k != 27:
        image = gen.get_next_image()
        print(image.shape)
        output_image = eval(encoder, decoder, image)
        cv2.imshow("style transfer", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        k = cv2.waitKey(3)

    cv2.destroyAllWindows()


def load_model(model_ckpt_dir):
    encoder = make_encoder_model()
    decoder = make_decoder_model((40, 40, 256))

    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                     decoder=decoder)

    if os.path.isdir(model_ckpt_dir):
        files = os.listdir(model_ckpt_dir)
        save_nb = (len(files) - 1) / 2
        last_checkpoint = os.path.join(model_ckpt_dir, "ckpt-{}".format(int(save_nb)))
    else:
        last_checkpoint = model_ckpt_dir
    logging.info("load model {}".format(last_checkpoint))
    checkpoint.restore(last_checkpoint)
    return encoder, decoder


def load_model_and_eval(model_ckpt_dir, images_dir, output_path):
    encoder, decoder = load_model(model_ckpt_dir)
    eval_dir(encoder, decoder, images_dir, output_path)


def load_model_and_eval_video(model_ckpt_dir, video_port):
    logging.info("load model {} and eval cam {}".format(model_ckpt_dir, video_port))
    encoder, decoder = load_model(model_ckpt_dir)
    eval_cam(encoder, decoder, video_port)
