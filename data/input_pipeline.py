import tensorflow.compat.v1 as tf


NUM_FRAMES = 4
SIZE = 256


class Pipeline:

    def __init__(self, filenames, is_training, batch_size):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size: an integer.
        """
        self.is_training = is_training

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.shuffle(len(filenames)) if is_training else dataset

        def get_subdataset(f):
            dataset = tf.data.TFRecordDataset(f)
            dataset = dataset.window(NUM_FRAMES, shift=1, drop_remainder=True)
            dataset = dataset.flat_map(lambda x: x.batch(NUM_FRAMES, drop_remainder=True))
            dataset = dataset.map(self.parse_and_preprocess)
            dataset = dataset.shuffle(1000) if is_training else dataset
            return dataset

        dataset = dataset.flat_map(get_subdataset)
        dataset = dataset.shuffle(10000) if is_training else dataset
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat(None if is_training else 1)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # tf.data.experimental.AUTOTUNE

        self.dataset = dataset

    def parse_and_preprocess(self, examples):
        """
        Returns:
            a uint8 tensor with shape [NUM_FRAMES, SIZE, SIZE, 2].
        """

        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string)
        }
        images_and_labels = []

        for i in range(NUM_FRAMES):
            parsed_features = tf.parse_single_example(examples[i], features)
            image = tf.image.decode_jpeg(parsed_features['image'], channels=1)
            labels = tf.image.decode_png(parsed_features['labels'], channels=1)
            images_and_labels.append(tf.concat([image, labels], axis=2))

        x = tf.stack(images_and_labels, axis=0)
        # it has shape [NUM_FRAMES, h, w, 2]

        if not self.is_training:
            shape = tf.shape(x)
            h, w = shape[1], shape[2]
            offset_height = (h - SIZE) // 2
            offset_width = (w - SIZE) // 2
            x = tf.image.crop_to_bounding_box(x, offset_height, offset_width, SIZE, SIZE)
        else:
            do_flip = tf.less(tf.random.uniform([]), 0.5)
            x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)
            x = tf.image.random_crop(x, [NUM_FRAMES, SIZE, SIZE, 2])

        return x
