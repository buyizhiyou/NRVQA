import argparse
import glob
import json
import os
import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from progressbar import *
from scipy import io

import imquality
import tensorflow as tf
from imquality import datasets
from utils import *

# ignore wanrnings
warnings.filterwarnings("ignore")
# arguments parser
parser = argparse.ArgumentParser(description="evaluate video quality.")
parser.add_argument("--category", help="video dir")
args = parser.parse_args()
category = args.category
vdir = "testdata/"+category
dirs = os.listdir(vdir)
dirs = natsorted(dirs)
print(dirs)


def image_preprocess(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image_low = gaussian_filter(image, 16, 7/6)
    image_low = rescale(image_low, 1/4, method=tf.image.ResizeMethod.BICUBIC)
    image_low = tf.image.resize(image_low, size=image_shape(
        image), method=tf.image.ResizeMethod.BICUBIC)

    return image-tf.cast(image_low, image.dtype)


input = tf.keras.Input(shape=(None, None, 1),
                       batch_size=1, name='original_image')
f = tf.keras.layers.Conv2D(48, (3, 3), name='Conv1',
                           activation='relu', padding='same')(input)
f = tf.keras.layers.Conv2D(48, (3, 3), name='Conv2',
                           activation='relu', padding='same', strides=(2, 2))(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv3',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv4',
                           activation='relu', padding='same', strides=(2, 2))(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv5',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv6',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(128, (3, 3), name='Conv7',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(128, (3, 3), name='Conv8',
                           activation='relu', padding='same')(f)

v = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(f)
h = tf.keras.layers.Dense(128, activation='relu')(v)
h = tf.keras.layers.Dense(1)(h)
# subjective_error_model = tf.keras.Model(
#     input, h, name='subjective_error_model')
# subjective_error_model.load_weights("models/")


subjective_error_model = tf.saved_model.load("models2/")
infer = subjective_error_model.signatures['serving_default']

# progressbar config
widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=len(dirs)).start()

data = {}
for i, d in enumerate(dirs):
    pbar.update(i+1)

    video_path = os.path.join(vdir, d)
    if not os.path.isdir(video_path):
        continue

    print("process:", d)
    files = glob.glob(video_path+"/*.mp4")
    files = natsorted(files)[1:]
    if len(files) == 0:
        continue

    scores = []
    for f in files:
        cap = cv2.VideoCapture(f)
        i = 0
        values = []
        while(1):
            ret, frame = cap.read()
            if frame is None:
                print("end")
                break
            i = i+1
            if i % 10 != 0:
                continue
            frame = np.expand_dims(frame, axis=0)
            im = tf.constant(frame)
            im = image_preprocess(im)
            out = infer(im)
            value = out['dense_1'][0][0].numpy()
            # import pdb
            # pdb.set_trace()
            # value = subjective_error_model.predict(im)[0][0]
            values.append(value)
            print(value)

        mean_score = np.mean(values)
        scores.append(mean_score)
    data[d] = scores

io.savemat("results2/"+category+"_diqa.mat", mdict=data)
