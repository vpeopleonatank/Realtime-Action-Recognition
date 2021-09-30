import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import time
import os
import simplejson

LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"

]
DATASET_PATH = "data/RNN-HAR-2D-Pose-database/"

X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32 # 32 timesteps per series

# Load the networks inputs

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)

    X_ = np.array(np.split(X_,blocks))

    return X_

# Load the networks outputs

def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # for 0-based indexing
    return y_

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
#print X_test

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)
# proof that it actually works for the skeptical: replace labelled classes with random classes to train on
#for i in range(len(y_train)):
#    y_train[i] = randint(0, 5)

#%%

def main(filepath):
    IMAGE_WIDTH, IMAGE_HEIGHT = 480, 640
    train_data = []
    clip_cnt = 0
    images_cnt = 0
    for x, y in zip(X_train, y_train):
        clip_cnt += 1
        for joints in x:
            data = []
            images_cnt += 1
            data.append(y[0].item())
            data.append(clip_cnt)
            data.append(images_cnt)
            data.append(LABELS[y[0]-1])
            data.append("NaN")
            for i, p in enumerate(joints):
                if i % 2 == 0:
                    data.append((p / IMAGE_WIDTH).item())
                else:
                    data.append((p / IMAGE_HEIGHT).item())
            train_data.append(data)
        break

    with open(filepath, 'w') as f:
        simplejson.dump(train_data, f)

if __name__ == "__main__":
    main("data/new_skeleton_data.txt")
