from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def random_forests_model(train_feature_set,train_label_set,test_feature_set,test_label_set):
    solver = ['newton-cg','lbfgs','sag']
    solver = ['lbfgs']
    results = {}
    final_coef = []
    final_result = []
    for i in solver:
        clf = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=100).fit(train_feature_set,train_label_set)
        results[i] = {}
        #results[i]['coef'] = clf.coef_
        preds = clf.predict(test_feature_set)
        results[i]['rmse'] = mean_squared_error(preds,test_label_set) ** 0.5
        results[i]['accuracy'] = clf.score(test_feature_set,test_label_set)
    return results

def get_latent_dataset(file_name):
    final_dataset = pd.read_csv(file_name)
    #final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    feature_dataset = final_dataset[['location','working_hours','parking','business_infra_features','business_extra_features','business_payments','accessibility','review_count']]
    label_dataset = final_dataset[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(label_dataset_encoded)
    train_feature_set,test_feature_set,train_label_set,test_label_set = train_test_split(feature_dataset, label_dataset_encoded, test_size = 0.2)

    return train_feature_set,train_label_set,test_feature_set,test_label_set

def get_all_features(file_name):
    final_dataset = pd.read_csv(file_name)
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    feature_dataset = final_dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking','pos_count','neg_count','checkin_count']]
    label_dataset = final_dataset[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    train_feature_set = feature_dataset.iloc[0:180000]
    test_feature_set = feature_dataset.iloc[180000:188594]

    train_label_set = label_dataset_encoded[0:180000]
    test_label_set = label_dataset_encoded[180000:188594]

    return train_feature_set,train_label_set,test_feature_set,test_label_set


final_results = {}
#train_feature_set,train_label_set,test_feature_set,test_label_set = get_latent_dataset('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_factoranalysis.csv')
#final_results['latent_dataset'] = random_forests_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

train_feature_set,train_label_set,test_feature_set,test_label_set = get_all_features('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_combined.csv')
final_results['all_features'] = random_forests_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

print(final_results)
