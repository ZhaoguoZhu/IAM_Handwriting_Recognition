import numpy as np


def normalize_pixels(train, test):
    train_scale = train.astype('float32')
    test_scale = test.astype('float32')
    train_scale = train_scale / 255.0
    test_scale = test_scale / 255.0
    return train_scale, test_scale
