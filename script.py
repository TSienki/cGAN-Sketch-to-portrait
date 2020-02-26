import argparse
from os import path

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from helpers.image import load_imgs, resize_imgs, normalize, IMG_WIDTH, IMG_HEIGHT


def load_images(image_path, sketch_path):
    img, sketch = load_imgs(image_path, sketch_path)
    img, sketch = resize_imgs(img, sketch, IMG_HEIGHT, IMG_WIDTH)
    img, sketch = normalize(img, sketch)

    return img, sketch


def main(sketch_path, save_path):
    model = load_model("models\\generator_model.h5")
    img, sketch = load_images(sketch_path, sketch_path)
    sketch = tf.reshape(sketch, (1, *sketch.shape))
    prediction = model(sketch, training=True)
    prediction = prediction.numpy() * 0.5 + 0.5
    plt.imsave(save_path, prediction[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sketch", help="sketch path")
    parser.add_argument("path", help="path to save generated file")
    args = parser.parse_args()
    main(args.sketch, args.path)
