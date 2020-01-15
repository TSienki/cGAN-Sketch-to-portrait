import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize_imgs(img, sketch, height, width):
    img = tf.image.resize(img, [height, width])
    sketch = tf.image.resize(sketch, [height, width])

    return img, sketch

def normalize(img, sketch):
    img = (img / 127.5) - 1
    sketch = (sketch / 127.5) - 1

    return img, sketch

def random_crop(img, sketch):
    stacked_image = tf.stack([img, sketch], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

@tf.function()
def random_jitter(img, sketch):
    # resizing to 286 x 286 x 3
    img, sketch = resize_imgs(img, sketch, 286, 286)

    # randomly cropping to 256 x 256 x 3
    img, sketch = random_crop(img, sketch)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        img = tf.image.flip_left_right(img)
        sketch = tf.image.flip_left_right(sketch)

    return img, sketch


def load_imgs(img_path, sketch_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    sketch = tf.io.read_file(sketch_path)

    img = tf.image.decode_png(img, channels=3)
    sketch = tf.image.decode_png(sketch, channels=3)

    img = tf.cast(img, tf.float32)
    sketch = tf.cast(sketch, tf.float32)

    return img, sketch
