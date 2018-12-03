# import the necessary packages
#from config import tiny_imagenet_config as config
#from pyimagesearch.preprocessing import ImageToArrayPreprocessor
#from pyimagesearch.preprocessing import SimplePreprocessor
#from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
#from pyimagesearch.io import HDF5DatasetGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from pyimagesearch.deepergooglenet import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from imutils import paths
import random
import cv2
import os
import keras.backend as K
import argparse
import json
import numpy as np
import operator
from collections import OrderedDict

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
ap.add_argument("-m", "--model", type=str,required=True,
        help="path to model")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []

# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images(args["dataset"]))

print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# update the learning rate
print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
K.set_value(model.optimizer.lr, 1e-5)
print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

positive_images = {}
negative_images = {}

# loop over the input images
for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255
        image = image.reshape((1,28,28,3))
        pred_class = model.predict_classes(image)	
        pred_prob = model.predict(image)
        image_id = imagePath.split(os.path.sep)[-1]
        if pred_class == 1:
                positive_images[image_id] = pred_prob[0][1]
        else:
                negative_images[image_id] = pred_prob[0][0]

sorted_p = sorted(positive_images.items(),key = operator.itemgetter(1))
sorted_n = sorted(negative_images.items(),key = operator.itemgetter(1))

pos_items = OrderedDict(sorted(positive_images.items(), 
                                  key=lambda kv: kv[1], reverse=True))
neg_items = OrderedDict(sorted(negative_images.items(), 
                                  key=lambda kv: kv[1], reverse=True))
print(len(positive_images.keys()))
print(len(negative_images.keys()))
print(pos_items)
print(neg_items)
