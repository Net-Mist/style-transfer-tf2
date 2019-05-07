import tensorflow as tf


def _discriminator_loss(logits, labels):
    """sce_criterion
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def abs_criterion(logits, target):
    """absolute criterion or L1 norm
    """
    return tf.reduce_mean(tf.abs(logits - target))


def mse_criterion(logits, target):
    return tf.reduce_mean((logits - target) ** 2)


def discriminator_loss(discriminate_encoded_picture, discriminate_picture, discriminate_art):
    """
    When training the discriminator, it needs to lean to classify the original art as 1 and the rest as 0
    Args:
        discriminate_encoded_picture:
        discriminate_picture:
        discriminate_art:

    Returns:

    """
    art_loss = 0
    picture_loss = 0
    encoded_picture_loss = 0
    for pred in discriminate_art:
        art_loss += _discriminator_loss(pred, tf.ones_like(pred))
    for pred in discriminate_picture:
        picture_loss += _discriminator_loss(pred, tf.zeros_like(pred))
    for pred in discriminate_encoded_picture:
        encoded_picture_loss += _discriminator_loss(pred, tf.zeros_like(pred))

    global_loss = art_loss + picture_loss + encoded_picture_loss

    logs = {
        "discriminator_loss/1_art": art_loss,
        "discriminator_loss/2_picture": picture_loss,
        "discriminator_loss/3_encoded": encoded_picture_loss,
        "discriminator_loss/4_global": global_loss
    }

    return global_loss, logs


def discriminator_acc(discriminate_encoded_picture, discriminate_picture, discriminate_art):
    art = []
    picture = []
    encoded = []
    for pred in discriminate_art:
        art.append(tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)), dtype=tf.float32)))
    for pred in discriminate_picture:
        picture.append(tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)), dtype=tf.float32)))
    for pred in discriminate_encoded_picture:
        encoded.append(tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)), dtype=tf.float32)))
    global_accuracy = tf.reduce_mean(art + picture + encoded)

    logs = {
        "discriminator_acc/1_art": tf.reduce_mean(art),
        "discriminator_acc/2_picture": tf.reduce_mean(picture),
        "discriminator_acc/3_encoded": tf.reduce_mean(encoded),
        "discriminator_acc/4_global": tf.reduce_mean(global_accuracy)
    }

    return global_accuracy, logs


def generator_loss(disc_output, transformed_input_image, input_features, transformed_output_image, output_features, img_loss_weight=100., feature_loss_weight=100.,
                   generator_weight=1.):
    losses = []
    for pred in disc_output:
        losses.append(_discriminator_loss(pred, tf.ones_like(pred)))

    generator_global_loss = tf.add_n(losses) * generator_weight

    # Image loss.
    img_loss = mse_criterion(transformed_output_image, transformed_input_image) * img_loss_weight

    # Features loss.
    feature_loss = abs_criterion(output_features, input_features) * feature_loss_weight

    global_loss = img_loss + feature_loss + generator_global_loss

    logs = {
        "generator_loss/1_feature": feature_loss,
        "generator_loss/2_disc": generator_global_loss,
        "generator_loss/3_transformer": img_loss,
        "generator_loss/4_global": global_loss,
    }

    return global_loss, logs


def generator_acc(disc_output):
    accuracies = []
    for pred in disc_output:
        accuracies.append(tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)), dtype=tf.float32)))

    generator_global_accuracy = tf.reduce_mean(accuracies)

    logs = {"generator_acc/global": generator_global_accuracy}

    return generator_global_accuracy, logs


def cartoon_content_loss(picture, generated, vgg_network):
    """Loss defined in cartoon gan paper

    The content loss is defined by using a pretrained vgg network to extract feature map from
    the images. By comparing the features with an L1 norm, we defined a distances between the images

    Args:
        picture: image without style applied
        generated: image after the style was applied
        vgg_network: network to use to extract feature map to compare both images

    Returns: loss between the two images

    """
    encoded_picture = vgg_network(picture)
    encoded_generated = vgg_network(generated)
    loss = abs_criterion(encoded_generated, encoded_picture)
    return loss


def cartoon_adversarial_loss(discriminate_cartoon, discriminate_smooth, discriminate_generated):
    loss = 0
    loss += _discriminator_loss(discriminate_cartoon, tf.zeros_like(discriminate_cartoon))
    loss += _discriminator_loss(discriminate_smooth, tf.ones_like(discriminate_smooth))
    loss += _discriminator_loss(discriminate_generated, tf.ones_like(discriminate_generated))
    return loss


def cartoon_generator_loss(discriminate_generated):
    return _discriminator_loss(discriminate_generated, tf.zeros_like(discriminate_generated))
