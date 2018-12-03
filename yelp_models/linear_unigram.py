
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error

def logsictic_model(train_feature_set,train_label_set,test_feature_set,test_label_set):
    solver = ['newton-cg','lbfgs','sag']
    solver = ['lbfgs']
    results = {}
    final_coef = []
    final_result = []
    for i in solver:
        clf = LinearRegression().fit(train_feature_set,train_label_set)
        results[i] = {}
        results[i]['coef'] = clf.coef_
        preds = clf.predict(test_feature_set)
        results[i]['rmse'] = mean_squared_error(preds,test_label_set) ** 0.5
        results[i]['accuracy'] = clf.score(test_feature_set,test_label_set)
    return results

def get_latent_dataset(file_name):
    final_dataset = pd.read_csv(file_name)
    final_dataset = final_dataset.sample(frac=1)

    feature_dataset = final_dataset[['location','working_hours','parking','business_infra_features','business_extra_features','business_payments','accessibility','review_count']]
    label_dataset = final_dataset[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    train_feature_set = feature_dataset.iloc[0:180000]
    test_feature_set = feature_dataset.iloc[180000:188594]

    train_label_set = label_dataset_encoded[0:180000]
    test_label_set = label_dataset_encoded[180000:188594]

    return train_feature_set,train_label_set,test_feature_set,test_label_set

def get_all_features(file_name):
    final = pd.read_csv(file_name)
    final_dataset = final.drop(['Unnamed: 0'], axis=1)
    final_dataset = final_dataset.drop(['business_id','latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking','checkin_count'], axis=1)
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    print('dropping')
#     final_dataset.dropna(inplace=True)
    print('done drop')
#     feature_dataset = final_dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking','pos_count','neg_count','checkin_count', 'text_length', 'count_punctuation']]
    feature_dataset = final_dataset.drop(['stars'], axis=1)
    label_dataset = final[['stars']]
    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)
    print('getting sets')
    train_feature_set = feature_dataset.iloc[0:180000]
    test_feature_set = feature_dataset.iloc[180000:188594]

    train_label_set = label_dataset_encoded[0:180000]
    test_label_set = label_dataset_encoded[180000:188594]

    return train_feature_set,train_label_set,test_feature_set,test_label_set


final_results = {}
#train_feature_set,train_label_set,test_feature_set,test_label_set = get_latent_dataset('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_factoranalysis.csv')
#final_results['latent_dataset'] = logsictic_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

train_feature_set,train_label_set,test_feature_set,test_label_set = get_all_features('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/features.csv')
final_results['all_features'] = logsictic_model(train_feature_set,train_label_set,test_feature_set,test_label_set)

print(final_results)

