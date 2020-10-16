import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import PIL

AUTO = tf.data.experimental.AUTOTUNE


def show_dataset(thumb_size, cols, rows, ds):
    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size * cols + (cols - 1),
                                             thumb_size * rows + (rows - 1)))
    for idx, data in enumerate(iter(ds)):
        img, target_or_imgid = data
        ix = idx % cols
        iy = idx // cols
        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        mosaic.paste(img, (ix * thumb_size + ix,
                           iy * thumb_size + iy))
    mosaic.show()


def map_function(img, imgname_or_label):
    return prepare_image(img), tf.cast(imgname_or_label, tf.int32)


def get_dataset(files, config, augment=False, shuffle=False, repeat=False,
                labeled=True, return_image_names=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    #ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    ds = ds.map(map_function,
                num_parallel_calls=AUTO)

    ds = ds.batch(160,drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def prepare_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [256,256,3])
    return img


def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    np.shape(example['image'])
    print(example['image'])
    return example['image'], example['target']