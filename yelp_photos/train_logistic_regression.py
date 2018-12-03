from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from utils import read_data, plot_data, plot_decision_function
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import csv
import os
import random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-f1","--positive_file_name",required=True, help="Path to the positive features")
ap.add_argument("-f2","--negative_file_name",required=True, help="Path to negative features")
ap.add_argument("-m","--model",required=True, help="Path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

positive_file    = args["positive_file_name"]
negative_file    = args["negative_file_name"]

positive = csv.reader(open(positive_file,'r'))
negative = csv.reader(open(negative_file,'r'))

X = []
Y = []

i = 0
for row in positive:
	if i==0:
		i+=1
		continue
	X.append(row[1:])
	Y.append(1)
	i+=1
	if i==5101:
		break

print("\nMeta information is as follows \n")
print("No of rows in Positive file are : "+str(i))
print("Size of X is : "+str(len(X)))
print("Size of Y is : "+str(len(Y)))

i = 0
for row in negative:
        if i==0:
                i+=1
                continue
        X.append(row[1:])
        Y.append(0)
        i+=1	

print("No of rows in Negative file are : "+str(i))
print("Size of X is : "+str(len(X)))
print("Size of Y is : "+str(len(Y)))

cut_off = int(len(X)*0.8)

X, Y = shuffle(X,Y)

# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

print("Type of X_train is : "+str(type(X_train)))

X_train = np.asarray(X_train, dtype=float, order=None)
y_train = np.asarray(y_train, dtype =float)
X_test  = np.asarray(X_test, dtype=float)
y_test  = np.asarray(y_test, dtype=float) 
print("Type of X_train is : "+str(type(X_train)))
print(X_train.shape)
print(y_train.shape)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(),params,cv=3,n_jobs=args["jobs"])
model.fit(X_train, y_train)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(X_test)
print(classification_report(y_test, preds,
	target_names=["Negative","Positive"]))

# serialize the model to disk
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()
