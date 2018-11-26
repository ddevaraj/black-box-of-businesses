from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def neural_model(train_feature_set,train_label_set,test_feature_set,test_label_set):
    solver = ['sgd', 'adam', 'lbfgs']
    results = {}
    final_coef = []
    final_result = []
    for i in solver:
        clf = MLPClassifier(solver=i, learning_rate_init = 0.001,alpha=1e-5,hidden_layer_sizes=(15,10), random_state=1,max_iter=10000).fit(train_feature_set,train_label_set)
        results[i] = {}
        #results[i]['coef'] = clf.coefs_
        results[i]['accuracy'] = clf.score(test_feature_set,test_label_set)
    return results

def get_latent_dataset(file_name):
    final_dataset = pd.read_csv(file_name)
    #final_dataset = final_dataset.sample(frac=1)
    #final_dataset.fillna(final_dataset.mean(), inplace=True)
    feature_dataset = final_dataset[['location','working_hours','parking','business_infra_features','business_extra_features','business_payments','accessibility','review_count']]
    label_dataset = final_dataset[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(label_dataset_encoded)
    train_feature_set,test_feature_set,train_label_set,test_label_set = train_test_split(feature_dataset, y, test_size = 0.1)

    return train_feature_set,train_label_set,test_feature_set,test_label_set

def get_all_features(file_name):
    final_dataset = pd.read_csv(file_name)
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    feature_dataset = final_dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking']][0:100]
    label_dataset = final_dataset[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(label_dataset_encoded)
    train_feature_set,test_feature_set,train_label_set,test_label_set = train_test_split(feature_dataset, y, test_size = 0.2)

    return train_feature_set,train_label_set,test_feature_set,test_label_set

final_results = {}
#train_feature_set,train_label_set,test_feature_set,test_label_set = get_latent_dataset('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_factoranalysis.csv')
#final_results['latent_dataset'] = logsictic_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

train_feature_set,train_label_set,test_feature_set,test_label_set = get_all_features('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business.csv')
final_results['all_features'] = neural_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

print(final_results)