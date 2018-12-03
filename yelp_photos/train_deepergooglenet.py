# USAGE
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_25.hdf5 --start-epoch 25

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-ml", "--model_load", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-m", "--model", type=str,required=True,
        help="path to model")
ap.add_argument("-p", "--plot", required=True,
        help="path to the output loss/accuracy plot")
ap.add_argument("-o", "--output", required=True,
        help="path to output directory (logs, plots, etc.)")
ap.add_argument("-w", "--weights", required=True,
        help="path to output directory (logs, plots, etc.)")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "Positive" else 0
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model_load"] is None:
	print("[INFO] compiling model...")
	model = DeeperGoogLeNet.build(width=64, height=64, depth=3,
		classes=2, reg=0.0002)
	opt = Adam(1e-3)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(
        os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(
        os.getpid())])
fname = os.path.sep.join([args["weights"],
        "weights-{epoch:03d}-{val_acc:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_acc", mode="max",
        save_best_only=True, verbose=1)
callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5,
		startAt=args["start_epoch"]),
	TrainingMonitor(figPath, jsonPath,
		startAt=args["start_epoch"]),checkpoint]

# train the network
H = model.fit_generator(
	aug.flow(trainX,trainY,batch_size=128),#trainGen.generator(),
	steps_per_epoch= len(trainX) // 128, #trainGen.numImages // 64,
	validation_data=(testX,testY), #valGen.generator(),
	validation_steps= len(testX) // 128, #valGen.numImages // 64,
	epochs=25,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 25), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 25), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 25), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 25), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
