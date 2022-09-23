import tensorflow as tf

import os
import pathlib
import time
import datetime
import glob
import random
import subprocess

import numpy as np
from matplotlib import pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

BUFFER_SIZE = 500
BATCH_SIZE = 4
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Fix seeds
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#### DIRECTORY UTILITIES ####


def directory_should_exist(*args):
    dir = os.path.join(*args)
    if not os.path.isdir(dir):
        raise Exception("Path '{}' is not a directory.".format(dir))
    return dir


def ensure_directory(*args):
    dir = os.path.join(*args)
    try:
        os.makedirs(dir)
    except OSError as err:
        if err.errno != 17:
            raise err
    return dir


#### IMAGE PROCESSING ####


def load_image(fname):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(fname)
    image = tf.io.decode_jpeg(image)

    # Split the image tensor into two tensors:
    # - The left (target) image
    # - The right (source) image
    image_width = tf.shape(image)[1]
    image_width = image_width // 2
    input_image = image[:, image_width:, :]
    real_image = image[:, :image_width, :]

    # If needed you can flip the direction of the training here.
    # real_image, input_image = input_image, real_image

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def normalize_images(input_image, real_image):
    # Normalize the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def resize_image(input_image, real_image, width, height):
    # Note that the order of width/height in tensorflow is reversed:
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.BICUBIC
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.BICUBIC
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


def random_jitter(input_image, real_image):
    # Resize the image to 572 x 572
    input_image, real_image = resize_image(
        input_image, real_image, IMAGE_WIDTH + 60, IMAGE_HEIGHT + 60
    )
    # Randomly crop the image back to 512x512
    input_image, real_image = random_crop(input_image, real_image)
    # Randomly flip the image
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


def process_image(fname):
    input_image, real_image = load_image(fname)
    input_image, real_image = normalize_images(input_image, real_image)
    input_image, real_image = random_jitter(input_image, real_image)
    return input_image, real_image


#### GENERATOR ####


def make_downsample_block(filters, size, apply_batch_norm=True):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    block = tf.keras.Sequential()
    block.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_batch_norm:
        block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.LeakyReLU())
    return block


def make_upsample_block(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.2)
    block = tf.keras.Sequential()
    block.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    block.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block


def make_generator():
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])
    down_stack = [
        make_downsample_block(32, 4, apply_batch_norm=False),  # (?, 256, 256,   32)
        make_downsample_block(64, 4),  # (?, 128, 128,   64)
        make_downsample_block(128, 4),  # (?,  64,  64,  128)
        make_downsample_block(256, 4),  # (?,  32,  32,  256)
        make_downsample_block(512, 4),  # (?,  16,  16,  512)
        make_downsample_block(512, 4),  # (?,   8,   8,  512)
        make_downsample_block(512, 4),  # (?,   4,   4,  512)
        make_downsample_block(512, 4),  # (?,   2,   2,  512)
        make_downsample_block(512, 4),  # (?,   1,   1,  512)
    ]

    up_stack = [
        make_upsample_block(512, 4, apply_dropout=True),  # (?,   2,   2, 1024)
        make_upsample_block(512, 4, apply_dropout=True),  # (?,   4,   4, 1024)
        make_upsample_block(512, 4, apply_dropout=True),  # (?,   8,   8, 1024)
        make_upsample_block(512, 4),  # (?,  16,  16, 1024)
        make_upsample_block(256, 4),  # (?,  32,  32,  512)
        make_upsample_block(128, 4),  # (?,  64,  64,  256)
        make_upsample_block(64, 4),  # (?, 128, 128,  128)
        make_upsample_block(32, 4),  # (?, 256, 256,   64)
    ]

    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # (?, 512, 512, 3)
    last = tf.keras.layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target, _lambda=100):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (_lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


#### DISCRIMINATOR ####


def make_discriminator():
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    input_image = tf.keras.layers.Input(shape=[512, 512, 3], name="input_image")
    target_image = tf.keras.layers.Input(shape=[512, 512, 3], name="target_image")
    x = tf.keras.layers.concatenate([input_image, target_image])  # (?, 512, 512,   6)
    down1 = make_downsample_block(32, 4, False)(x)  # (?, 256, 256,  32)
    down2 = make_downsample_block(64, 4)(down1)  # (?, 128, 128,  64)
    down3 = make_downsample_block(128, 4)(down2)  # (?,  64,  64, 128)
    down4 = make_downsample_block(256, 4)(down3)  # (?,  32,  32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (?,  34,  34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (?,  31,  31, 512)
    batch_norm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batch_norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (?,  33,  33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (?, 30, 30, 1)
    return tf.keras.Model(inputs=[input_image, target_image], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_loss = real_loss + generated_loss
    return total_loss


#### TRAINING ####


def save_image(filename, tensor, output_images_dir):
    normalized = tensor * 0.5 + 0.5
    image_tensor = tf.cast(normalized * 255, tf.uint8)
    tf.io.write_file(
        os.path.join(output_images_dir, filename), tf.io.encode_jpeg(image_tensor)
    )


def generate_images(model, test_input, target, output_images_dir, step=1):
    prediction = model(test_input, training=True)
    # We use the same input and target images, so only save them once.
    if step == 0:
        save_image(f"input-{step:05d}.jpg", test_input[0], output_images_dir)
        save_image(f"target-{step:05d}.jpg", target[0], output_images_dir)
    save_image(f"output-{step:05d}.jpg", prediction[0], output_images_dir)


def _train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    summary_writer,
    step,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
        tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)


def main(
    dataset_dir,
    model_dir,
    checkpoint_dir,
    log_dir,
    output_images_dir,
    glob_pattern,
    steps,
):
    train_dataset = tf.data.Dataset.list_files(os.path.join(dataset_dir, glob_pattern))
    train_dataset = train_dataset.map(
        process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    generator = make_generator()
    discriminator = make_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=5
    )
    checkpoint_manager.restore_or_initialize()

    summary_writer = tf.summary.create_file_writer(
        log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    example_input, example_target = next(iter(train_dataset.take(1)))
    for step, (input_image, target) in train_dataset.repeat().take(steps).enumerate():
        _train_step(
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            input_image,
            target,
            summary_writer,
            step,
        )
        # Plot a dot every 10 steps
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)
        # Generate images every 10 steps
        if step % 10 == 0:
            generate_images(
                generator,
                example_input,
                example_target,
                output_images_dir,
                step=step // 10,
            )
        # Save checkpoint every 1000 steps
        if (step + 1) % 1000 == 0:
            checkpoint_manager.save()
    print("Training done.")

    checkpoint.save(file_prefix=checkpoint_prefix)
    generator.save(model_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="assets/faces_segmented")
    parser.add_argument("--training-dir", type=str, default="assets/v001")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--output-images-dir", type=str, default=None)
    parser.add_argument("--glob-pattern", type=str, default="*.jpg")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    model_dir = os.path.join(args.training_dir, "model")
    if args.model_dir is not None:
        model_dir = args.model_dir
    checkpoint_dir = os.path.join(args.training_dir, "checkpoints")
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    log_dir = os.path.join(args.training_dir, "log")
    if args.log_dir is not None:
        log_dir = args.log_dir
    output_images_dir = os.path.join(args.training_dir, "images")
    if args.output_images_dir is not None:
        output_images_dir = args.output_images_dir

    main(
        dataset_dir=args.dataset_dir,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        output_images_dir=output_images_dir,
        glob_pattern=args.glob_pattern,
        steps=args.steps,
    )
