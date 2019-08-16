import os
import cv2
import numpy as np
from tqdm import tqdm


VIDEOS = 'data/videos/'
MIN_DIMENSION = 256


def get_counts(path):

    counts = np.zeros(256 * 256, dtype='int64')
    identifiers = np.arange(256 * 256)
    identifiers = identifiers.reshape(256, 256)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_dimension = min(width, height)
    scaler = MIN_DIMENSION/min_dimension
    new_size = (int(width * scaler), int(height * scaler))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame = frame[:, :, 1:]
            frame = frame.reshape(-1, 2)
            y, x = frame[:, 0], frame[:, 1]
            i, c = np.unique(identifiers[y, x], return_counts=True)
            counts[i] += c
        else:
            break

    cap.release()
    return counts


names = os.listdir(VIDEOS)
total_counts = np.zeros(256 * 256, dtype='int64')

for n in tqdm(names):

    path = os.path.join(VIDEOS, n)
    c = get_counts(path)
    total_counts += c

print('max count is', total_counts.max())
np.save('counts.npy', total_counts)
