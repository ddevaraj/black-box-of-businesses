import argparse
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import json
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-c","--file_name",required=True, help="Path to the conv features")
ap.add_argument("-m","--model",required=True, help="Path to output model")
ap.add_argument("-n","--negative",required=True, help="Path of ids of negative images")
ap.add_argument("-p","--positive",required=True, help="Path of ids of positive images")
ap.add_argument("-i","--iterations",required=True, help="No of iterations")
ap.add_argument("-j","--path_to_json",required=True, help="Path to json file")

args = vars(ap.parse_args())

i= 0

negative_data = []
negative_labels = []

positive_data = []
positive_labels = []

negative_photo_ids = []
positive_photo_ids = []
all_ids = []
all_id = []
remaining_ones = []
remaining_ones_ids = []

images_to_be_added_in_this_iteration = 50

train_accuracies = []
validation_accuracies = []

def create_list_of_negative_photo_ids(file_name):
        with open(file_name) as fp:
            line = fp.readline()
            while line:
                negative_photo_ids.append(line.strip())
                line = fp.readline()
            return len(negative_photo_ids)

def create_list_of_positive_photo_ids(file_name):
        with open(file_name) as fp:
            line = fp.readline()
            while line:
                positive_photo_ids.append(str(line.strip()))
                line = fp.readline()
            return len(positive_photo_ids)

no_of_iterations = int(args["iterations"])

count1 = 0
count2 = 0
count1+=create_list_of_negative_photo_ids(args["negative"])
count2+=create_list_of_positive_photo_ids(args["positive"])
total_count = count1 + count2
print("No of images in file 1 are : "+str(count1))
print("No of images in file 2 are : "+str(count2))
print("Total no of images are : "+str(total_count))

no_of_lines = 0
positive_ids = []
negative_ids = []
with open(args["file_name"]) as f:
	for line in f:
		if i==0:
			i+=1
			continue
		else:
			no_of_lines+=1
		        features = line.split(",")
                        if features[0][1:len(features[0])-1] in negative_photo_ids:
				negative_data.append(features[1:])
				negative_labels.append(0)
				negative_ids.append(features[0][1:len(features[0])-1])
			elif features[0][1:len(features[0])-1] in positive_photo_ids:
				positive_data.append(features[1:])
				positive_labels.append(1)
				positive_ids.append(features[0][1:len(features[0])-1])
			else:
				remaining_ones.append(features[1:])
				remaining_ones_ids.append(features[0][1:len(features[0])-1])
print("Positive photo ids length is : "+str(len(positive_photo_ids)))
print("Negative photo ids length is : "+str(len(negative_photo_ids)))

print("Len of positive data is : "+str(len(positive_data)))
print("Len of negative data is : "+str(len(negative_data)))

print("Len of remaining data is : "+str(len(remaining_ones)))
print("Len of remaining ones ids is :"+str(len(remaining_ones_ids)))

all_data = positive_data + negative_data + remaining_ones
all_id = positive_ids + negative_ids
print("Len of all ids 1 is : "+str(len(all_id)))
all_ids = all_id + remaining_ones_ids

print("Len of all data is : "+str(len(all_data)))
print("Len of all ids is : "+str(len(all_ids)))
positive_data = np.array(positive_data)
positive_labels = np.array(positive_labels)
negative_data = np.array(negative_data)
negative_labels = np.array(negative_labels)
remaining_ones = np.array(remaining_ones)

all_data = np.concatenate((positive_data,negative_data))
all_data = np.concatenate((all_data,remaining_ones))

positive_data,positive_labels = shuffle(positive_data,positive_labels)
negative_data,negative_labels = shuffle(negative_data,negative_labels)

train_data        = np.concatenate((positive_data[:201],negative_data[:201]),axis=0)
train_labels      = np.concatenate((positive_labels[:201],negative_labels[:201]),axis=0)
validation_data   = np.concatenate((positive_data[201:453],negative_data[201:243]),axis=0)
validation_labels = np.concatenate((positive_labels[201:453],negative_labels[201:243]),axis=0)
test_data         = np.concatenate((positive_data[453:705],negative_data[243:]),axis=0)
test_labels       = np.concatenate((positive_labels[453:705],negative_labels[243:]),axis=0)

print("#####################################")
print("")
print("Meta information is as follows ")
print("Positive Data shape is : "+str(positive_data.shape))
print("Positive labels shape is : "+str(positive_labels.shape))
print("Negative Data shape is : "+str(negative_data.shape))
print("Negative labels shape is : "+str(negative_labels.shape))
print("")
print("Train Data shape is : "+str(train_data.shape))
print("Train labels shape is : "+str(train_labels.shape))
print("Validation Data shape is : "+str(validation_data.shape))
print("Validation labels shape is : "+str(validation_labels.shape))
print("Test Data shape is : "+str(test_data.shape))
print("Test labels shape is : "+str(test_labels.shape))
print("")
print("#####################################")

print("No of lines in csv file are : "+str(no_of_lines))

# Split data to train and test on 80-20 ratio
#X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state=0)

print("the first id in all_ids is : "+str(all_ids[0]))
def train_an_svm(iteration_num):
	print("")
	print("Iteration : "+str(iteration_num))
	
	global train_data
	global train_labels
	global test_data
	global test_labels
	global validation_data
	global validation_labels
	global images_to_be_added_in_this_iteration
	global remaining_ones
	global remaining_ones_ids
	
	X_train, X_test, y_train, y_test = train_data,test_data,train_labels,test_labels
	
	print("Train data shape is : "+str(X_train.shape))
	print("Test data shape is : "+str(X_test.shape))	
	
	
	# Grid Search
	# Parameter Grid
	param_grid = {'C': [0.001,0.01,0.1, 1,2,3,10, 100], 'gamma': [1, 0.1, 0.01, 0.02,0.03,0.001, 0.00001, 10]}

	print("[INFO] Tuning hyper parameters")

	# Make grid search classifier
	clf_grid = GridSearchCV(svm.SVC(probability=True), param_grid, verbose=1)

	# Train the classifier
	clf_grid.fit(X_train,y_train)

	print("Best Parameters:\n", clf_grid.best_params_)
	print("Best Estimators:\n", clf_grid.best_estimator_)

	# evaluate the model
	print("[INFO] evaluating...")
	preds_proba = clf_grid.predict_proba(X_test)
	
	idx_num = 0
	max_in_pos = 0
	max_in_neg = 0
	top_pos = []
	top_neg = []
	for i in remaining_ones:
        	pred_pr = clf_grid.predict_proba(np.array(i).reshape(1,-1))
		idx = np.argmax(pred_pr)
		if idx==1:
			if pred_pr[0][idx]>max_in_pos:
				max_in_pos = pred_pr[0][idx]
			top_pos.append((pred_pr[0][idx],remaining_ones_ids[idx_num]))
		elif idx==0:
			if pred_pr[0][idx]>max_in_pos:
                        	max_in_neg = pred_pr[0][idx]
			top_neg.append((pred_pr[0][idx],remaining_ones_ids[idx_num]))
	
		idx_num+=1

	top_pos = sorted(top_pos, key=lambda tup: tup[0],reverse=True)
	top_neg = sorted(top_neg, key=lambda tup: tup[0],reverse=True)

	pos_to_be_added = []
	neg_to_be_added = []

	for i in range(images_to_be_added_in_this_iteration):
		prob,id = top_pos[i]
		pos_to_be_added.append(id)
		prob,id = top_neg[i]
        	neg_to_be_added.append(id)
		
	images_to_be_added_in_this_iteration = int(images_to_be_added_in_this_iteration*1.1)

	idx1 = 0
	ids_to_be_removed = []
	for i in remaining_ones_ids:
		if i in pos_to_be_added:
			train_data = np.concatenate((train_data,remaining_ones[idx1].reshape(1,-1)),axis=0)
			label = [1]
			label = np.array(label)
			train_labels = np.concatenate((train_labels,label),axis=0)
			ids_to_be_removed.append(idx1)
		elif i in neg_to_be_added:
			train_data = np.concatenate((train_data,remaining_ones[idx1].reshape(1,-1)),axis=0)
			label = [0]
                	label = np.array(label)
			train_labels = np.concatenate((train_labels,label),axis=0)
			ids_to_be_removed.append(idx1)
		idx1+=1
	print(len(top_pos))
	print(len(top_neg))
	
	print("###################################")
	print("")
	print("Train")
	preds = clf_grid.predict(X_train)
	print("Train accuracy is : "+str(accuracy_score(y_train,preds)))
	train_accuracies.append(accuracy_score(y_train,preds))
	print("")
	print("###################################")
	print("")
	
	print("###################################")
	print("")
	print("Validation")
	preds = clf_grid.predict(validation_data)
	print("Validation accuracy is : "+str(accuracy_score(validation_labels,preds)))
	validation_accuracies.append(accuracy_score(validation_labels,preds))
	print("")
	print("###################################")

	print("Max in pos : ",max_in_pos)
	print("Max in neg : ",max_in_neg)
	# serialize the model to disk
	print("[INFO] saving model...")
	f = open(args["model"], "wb")
	f.write(pickle.dumps(clf_grid.best_estimator_))
	f.close()

	print("Remaining ones length is : "+str(len(remaining_ones)))

	remaining_ones = np.delete(remaining_ones,ids_to_be_removed,axis=0)
	remaining_ones_ids = np.delete(remaining_ones_ids,ids_to_be_removed,axis=0)

	print("Remaining ones length is : "+str(len(remaining_ones)))
	
	if iteration_num%5==0 or iteration_num+1 == no_of_iterations:
		print("###################################")
        	print("")
		print("Test")
        	preds = clf_grid.predict(test_data)
        	print("Test accuracy is : "+str(accuracy_score(test_labels,preds)))
		print("")
		print("###################################")

	itr_num = 0
	if iteration_num+1 == no_of_iterations:
		business_ids_pos = {}
		business_ids_neg = {}
		photo_id_to_business_id = {}
		no_line = 0
		with open(args["path_to_json"]) as f:
			for line in f:
				no_line+=1
				data = json.loads(line)
				#if data["business_id"] not in business_ids_pos.keys():
				business_ids_pos[data["business_id"]] = 0
				#if data["business_id"] not in business_ids_neg.keys():
                                business_ids_neg[data["business_id"]] = 0
				photo_id_to_business_id[data["photo_id"]] = data["business_id"]
		print("No of lines parsed in json are : "+str(no_line))
		print("The length of photo_id_to_business_id is : "+str(len(photo_id_to_business_id)))
		print("[INFO] Done with json file")
		f = open("final_labels.txt","w")
		count=0
		no_of_keys_not_there = 0
		for row in all_data:
			#print(count)
			pred_pr = clf_grid.predict_proba(np.array(row).reshape(1,-1))
                	idx = np.argmax(pred_pr)
			'''
			if all_ids[itr_num] not in business_ids_pos.keys():
				no_of_keys_not_there+=1
				continue
			'''
			if idx == 1:
				x = business_ids_pos[photo_id_to_business_id[all_ids[itr_num]]]
				business_ids_pos[photo_id_to_business_id[all_ids[itr_num]]] = x+1
			else:
				x = business_ids_neg[photo_id_to_business_id[all_ids[itr_num]]]
                                business_ids_neg[photo_id_to_business_id[all_ids[itr_num]]] = x+1
			count+=1
			itr_num+=1
		print("No of keys not there are : "+str(no_of_keys_not_there))
		print("Count is : "+str(count))
		print("No of keys in business_ids_pos are : "+str(len(business_ids_pos.keys())))
		f.write("business_id,positive_photos,negative_photos\n")
                for key in business_ids_pos.keys():
			f.write(str(key)+","+str(business_ids_pos[key])+","+str(business_ids_neg[key]))
               		f.write("\n")
				
		
for i in range(no_of_iterations):
	train_an_svm(i)


iterations = []
for i in range(no_of_iterations):
	iterations.append(i)

plt.style.use("ggplot")
plt.figure()
plt.plot(iterations, train_accuracies, label="Train")
plt.plot(iterations, validation_accuracies, label="Val")
plt.title("Training and Validation accuracies SVM")
plt.legend()
plt.savefig("train_val_accuracies_svm.png")
plt.close()

