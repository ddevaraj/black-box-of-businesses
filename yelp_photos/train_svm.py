from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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

#print("Displaying data. Close window to continue")
# Plot data 
#plot_data(X_train, y_train, X_test, y_test)

print("Training SVM ...")
# make a classifier
clf = svm.SVC(C = 10.0, kernel='rbf', gamma=0.1)

# Train classifier
clf.fit(X_train, y_train)

# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)

#print("Displaying decision function. Close window to continue")
# Plot decision function on training and test data
#plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

print("[INFO] Tuning hyper parameters")

# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)

# Train the classifier
clf_grid.fit(X_train,y_train)

print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

#print("Displaying decision function for best estimator. Close window to continue.")
# Plot decision function on training and test data
#plot_decision_function(X_train, y_train, X_test, y_test, clf_grid)

# evaluate the model
print("[INFO] evaluating...")
preds = clf_grid.predict(X_test)
print(classification_report(y_test, preds,
	target_names=["Negative","Positive"]))

# serialize the model to disk
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()
