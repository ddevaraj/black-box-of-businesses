# USAGE
# python train_model.py --db ../datasets/animals/hdf5/features.hdf5 \
#	--model animals.cpickle
# python train_model.py --db ../datasets/caltech-101/hdf5/features.hdf5 \
#	--model caltech101.cpickle
# python train_model.py --db ../datasets/flowers17/hdf5/features.hdf5 \
#	--model flowers17.cpickle

# import the necessary packages
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
	help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

# Create a linear SVM classifier 
clf = svm.SVC(kernel='linear')

# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

print("[INFO] Tuning hyper parameters")

# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)

# Train the classifier
clf_grid.fit(db["features"][:i], db["labels"][:i])

print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

# evaluate the model
print("[INFO] evaluating...")
preds = clf_grid.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
	target_names=db["label_names"]))

# serialize the model to disk
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()
