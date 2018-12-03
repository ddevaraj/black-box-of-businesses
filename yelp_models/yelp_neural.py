from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def neural_model(train_feature_set,train_label_set,test_feature_set,test_label_set):
    solver = ['adam']
    learning_rate = [0.01]
    results = {}
    final_coef = []
    final_result = []
    for i in solver:
        for j in learning_rate:
            clf = MLPClassifier(solver=i, learning_rate_init = 0.001,alpha=1e-5,hidden_layer_sizes=(10,10,10), random_state=1,max_iter=10000).fit(train_feature_set,train_label_set)
            results[i] = {}
            #results[i]['coef'] = clf.coefs_
            preds = clf.predict(test_feature_set)
            results[i]['rmse'] = mean_squared_error(preds,test_label_set) ** 0.5
            results[i]['accuracy'] = clf.score(test_feature_set,test_label_set)
            print(i,j,results[i]['rmse'],results[i]['accuracy'])
    return results

def get_latent_dataset(file_name):
    final_dataset = pd.read_csv(file_name)
    final_dataset = final_dataset[['location','stars','working_hours','parking','business_infra_features','business_extra_features','business_payments','accessibility','review_count']]
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
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
    final_dataset = final_dataset[['latitude','stars','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking']]
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    feature_dataset = final_dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking']]
    label_dataset = final_dataset[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(label_dataset_encoded)
    train_feature_set,test_feature_set,train_label_set,test_label_set = train_test_split(feature_dataset, y, test_size = 0.2)

    return train_feature_set,train_label_set,test_feature_set,test_label_set

def get_all_features_reviews(file_name):
    final_dataset = pd.read_csv(file_name)
    final_dataset = final_dataset.drop('business_id',axis=1)
    final_dataset = final_dataset.drop('categories',axis=1)
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    #feature_dataset = final_dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking','pos_count','neg_count','checkin_count']] 
    label_dataset = final_dataset['stars']
    feature_dataset = final_dataset.drop('stars',axis=1)
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(label_dataset_encoded)

    train_feature_set,test_feature_set,train_label_set,test_label_set = train_test_split(feature_dataset, label_dataset_encoded, test_size = 0.2)

    return train_feature_set,train_label_set,test_feature_set,test_label_set

final_results = {}

'''
train_feature_set,train_label_set,test_feature_set,test_label_set = get_latent_dataset('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_factoranalysis.csv')
final_results['latent_dataset'] = neural_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

'''
train_feature_set,train_label_set,test_feature_set,test_label_set = get_all_features('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business.csv')
final_results['all_features'] = neural_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

train_feature_set,train_label_set,test_feature_set,test_label_set = get_all_features_reviews('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_temp.csv')
final_results['all_features_reviews'] = neural_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

outfile = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_model_results/yelp_neural.txt','w')
outfile.write(str(final_results))
