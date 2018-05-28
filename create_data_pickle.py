from common import load_track, FEELING
import numpy as np
from _pickle import dump
import os

TRACK_COUNT = 1000

def get_default_shape(dataset_path):
    tmp_features, _ = load_track(os.path.join(dataset_path,
        'Feel good/Feel good.00000.au'))
    return tmp_features.shape

def collect_data(dataset_path):

    default_shape = get_default_shape(dataset_path)
    x = np.zeros((TRACK_COUNT,) + default_shape, dtype=np.float32)
    y = np.zeros((TRACK_COUNT, len(FEELING)), dtype=np.float32)
    track_paths = {}

    for (genre_index, genre_name) in enumerate(FEELING):
        for i in range(TRACK_COUNT // len(FEELING)):
            file_name = '{}/{}.000{}.au'.format(genre_name,
                    genre_name, str(i).zfill(2))
            print ('Processing', file_name)
            path = os.path.join(dataset_path, file_name)
            track_index = genre_index  * (TRACK_COUNT // len(FEELING)) + i
            x[track_index], _ = load_track(path, default_shape)
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)

    return (x, y, track_paths)

if __name__ == '__main__':

    (x, y, track_paths) = collect_data('data/genres')

    data = {'x': x, 'y': y, 'track_paths': track_paths}
    with open('data/data.pkl', 'wb') as f:
        dump(data, f)
