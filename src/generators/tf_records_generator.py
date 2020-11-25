import tensorflow as tf
import tensorflow.keras.backend as K
import re
import numpy as np
import math
import PIL

AUTO = tf.data.experimental.AUTOTUNE


class DatasetGenerator:

    def __init__(self, config: dict):
        self.config = config


    def get_dataset(self,files,augment=False, repeat=False,
                    upsampling=False, patient_info=False):
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
        # ds = ds.cache()

        if repeat:
            ds = ds.repeat()

        if patient_info:
            ds = ds.map(self.read_labeled_tfrecord_with_patient_info, num_parallel_calls=AUTO)
            ds = ds.map(self.map_function_with_patient_info, num_parallel_calls=AUTO)
        else:
            ds = ds.map(self.read_labeled_tfrecord, num_parallel_calls=AUTO)
            ds = ds.map(self.map_function, num_parallel_calls=AUTO)

        if upsampling:
            ds = ds.flat_map(self.upsample_map)

        if augment:
            ds = ds.map(self.augmentation_map, num_parallel_calls=AUTO)

        ds = ds.shuffle(1024 * 8, seed=42)

        ds = ds.batch(self.config["batch_size"] * self.config["replicas"], drop_remainder=True)
        ds = ds.prefetch(AUTO)
        return ds


    def augmentation_map(self, input_data, target):
        img = input_data[0]
        img = self.transform(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.clip_by_value(img, 0, 1)
        return (img, input_data[1]), target

    def upsample_map(self, x, y):
        return tf.data.Dataset.from_tensors((x, y)).repeat(self.num_of_upsamples((x, y)))

    def num_of_upsamples(self, example):
        if example[1] == 1:
            return tf.cast(self.config["upsample_ratio"], tf.int64)
        else:
            return tf.cast(1, tf.int64)

    def map_function_with_patient_info(self, input_data, imgname_or_label):
        return (self.prepare_image(input_data[0]), self.prepare_patient_info(input_data[1])), tf.cast(imgname_or_label, tf.int32)

    def map_function(self, img, imgname_or_label):
        return self.prepare_image(img), tf.cast(imgname_or_label, tf.int32)

    def prepare_patient_info(self, patient):
        sex = tf.one_hot(patient[0], 2)
        age_aprox = tf.dtypes.cast(tf.reshape(patient[1], [1]), tf.float32)
        anatom_site = tf.one_hot(patient[2], 6)

        patient_info = tf.concat([sex, age_aprox, anatom_site], axis=0)
        return patient_info

    def prepare_image(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.reshape(img, [self.config["image_resolution"], self.config["image_resolution"], 3])
        return img


    def read_labeled_tfrecord_with_patient_info(self, example):
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
        return (example['image'], (example['sex'], example['age_approx'],
                                   example['anatom_site_general_challenge'])), example['target']

    def read_labeled_tfrecord(self, example):
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

    def get_mat(self, rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
        # returns 3x3 transformmatrix which transforms indicies

        # CONVERT DEGREES TO RADIANS
        rotation = math.pi * rotation / 180.
        shear = math.pi * shear / 180.

        def get_3x3_mat(lst):
            return tf.reshape(tf.concat([lst], axis=0), [3, 3])

        # ROTATION MATRIX
        c1 = tf.math.cos(rotation)
        s1 = tf.math.sin(rotation)
        one = tf.constant([1], dtype='float32')
        zero = tf.constant([0], dtype='float32')

        rotation_matrix = get_3x3_mat([c1, s1, zero,
                                       -s1, c1, zero,
                                       zero, zero, one])
        # SHEAR MATRIX
        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)

        shear_matrix = get_3x3_mat([one, s2, zero,
                                    zero, c2, zero,
                                    zero, zero, one])
        # ZOOM MATRIX
        zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero,
                                   zero, one / width_zoom, zero,
                                   zero, zero, one])
        # SHIFT MATRIX
        shift_matrix = get_3x3_mat([one, zero, height_shift,
                                    zero, one, width_shift,
                                    zero, zero, one])

        return K.dot(K.dot(rotation_matrix, shear_matrix),
                     K.dot(zoom_matrix, shift_matrix))

    def transform(self, image):
        # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        # output - image randomly rotated, sheared, zoomed, and shifted
        DIM = self.config["image_resolution"]
        XDIM = DIM % 2  # fix for size 331

        rot = 180.0 * tf.random.normal([1], dtype='float32')
        shr = 2.0 * tf.random.normal([1], dtype='float32')
        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 8.0
        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 8.0
        h_shift = 8.0 * tf.random.normal([1], dtype='float32')
        w_shift = 8.0 * tf.random.normal([1], dtype='float32')

        # GET TRANSFORMATION MATRIX
        m = self.get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

        # LIST DESTINATION PIXEL INDICES
        x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
        y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
        z = tf.ones([DIM * DIM], dtype='int32')
        idx = tf.stack([x, y, z])

        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = K.cast(idx2, dtype='int32')
        idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

        # FIND ORIGIN PIXEL VALUES
        idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
        d = tf.gather_nd(image, tf.transpose(idx3))

        img = tf.reshape(d, [DIM, DIM, 3])
        img = tf.clip_by_value(img, 0, 1)
        return img

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




















def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


