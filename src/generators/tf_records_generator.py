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


def get_dataset(files, config, augment=False, shuffle=False, repeat=False,
                labeled=True, return_image_names=True, patient_info=False):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    # ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if patient_info:
        ds = ds.map(read_labeled_tfrecord_with_patient_info, num_parallel_calls=AUTO)
        ds = ds.map(map_function_with_patient_info, num_parallel_calls=AUTO)
    else:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
        ds = ds.map(map_function, num_parallel_calls=AUTO)

    ds = ds.batch(config["batch_size"] * config["replicas"], drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def map_function_with_patient_info(input_data, imgname_or_label):
    return [prepare_image(input_data[0]), prepare_patient_info(input_data[1])], tf.cast(imgname_or_label, tf.int32)


def map_function(img, imgname_or_label):
    return prepare_image(img), tf.cast(imgname_or_label, tf.int32)


def prepare_patient_info(patient):
    patient_info = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if patient["sex"] == 0:
        patient_info[0] = 1
    elif patient["sex"] == 1:
        patient_info[1] = 1

    patient_info[2] = patient["age"]

    if patient["anatom_site_general_challenge"] != -1:
        patient_info[3 + patient["anatom_site_general_challenge"]] = 1
    patient_info = tf.cast(patient_info, tf.int32)
    patient_info = tf.reshape(patient_info, 9)
    return patient_info

def prepare_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [256, 256, 3])
    return img


def read_labeled_tfrecord_with_patient_info(example):
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
    return [example['image'], [example['sex'], example['age_approx'],
                               example['anatom_site_general_challenge']]], example['target']


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
    return example['image'], example['target']
