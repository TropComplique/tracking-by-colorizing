import io
import os
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import shutil
from tqdm import tqdm


VIDEOS_PATH = 'data/videos/'
OUTPUT_FOLDER = 'data/shards/'
MIN_DIMENSION = 256
FPS = 6

# ENCODER = np.load('encoder.npy')
# it has shape [256, 256] and type uint8


def get_tf_example(frame):
    """
    Arguments:
        frame: a numpy uint array with shape [h, w, 3],
            it represents an image in LAB color space.
    Returns:
        an instance of tf.Example.
    """
    h, w, c = frame.shape
    assert c == 3

    gray = frame[:, :, 0]  # shape [h, w]
    color = frame[:, :, 1:]  # shape [h, w, 2]

    color = color.reshape(-1, 2)  # shape [h * w, 2]
    a, b = color[:, 0], color[:, 1]
    labels = ENCODER[a, b]  # shape [h * w]
    labels = labels.reshape(h, w)

    encoded_gray = to_bytes(gray, 'jpeg')
    encoded_labels = to_bytes(labels, 'png')

    feature = {
        'image': bytes_feature(encoded_gray),
        'labels': bytes_feature(encoded_labels)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_bytes(array, image_format):
    image = Image.fromarray(array)
    b = io.BytesIO()
    image.save(b, format=image_format)
    return b.getvalue()


def write_frames(video_path, shard_path):

    cap = cv2.VideoCapture(video_path)
    writer = tf.python_io.TFRecordWriter(shard_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_dimension = min(width, height)
    scaler = MIN_DIMENSION/min_dimension
    new_size = (int(width * scaler), int(height * scaler))
    assert min(new_size) == MIN_DIMENSION

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        # this changes video frame rate
        # from 'fps' to 'FPS'
        take_frame = int(count % (fps/FPS)) == 0
        count += 1

        if ret:
            if take_frame:
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                example = get_tf_example(frame)
                writer.write(example.SerializeToString())
        else:
            break

    cap.release()
    writer.close()


def main():

    names = os.listdir(VIDEOS_PATH)[:10]
    names = sorted(names)
    print('Number of videos:', len(names))

    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.mkdir(OUTPUT_FOLDER)

    for i, n in tqdm(enumerate(names)):
        video_path = os.path.join(VIDEOS_PATH, n)
        shard_path = os.path.join(OUTPUT_FOLDER, f'shard-{i:04d}.tfrecords')
        write_frames(video_path, shard_path)


main()
