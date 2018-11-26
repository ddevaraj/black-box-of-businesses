from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import numpy as np
import pandas as pd
from gensim.models import FastText

final_headers = []
all_features_business = pd.read_csv("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business.csv")
#all_features_business = all_features_business[['latitude', 'longitude']]
#groups_headers = [location, working_hours, parking, business_infra_features, business_extra_features, business_payments, accessibility, no_reduction_factors]
groups_headers_dict = {'location':['latitude', 'longitude'],
                       'working_hours':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'is_open', 'Open24Hours'],
                       'parking':['BikeParking','BusinessParking'],
                       'business_infra_features':['OutdoorSeating', 'HasTV', 'WiFi'],
                       'business_extra_features':['Alcohol','HappyHour','Caters', 'CoatCheck',  'GoodForKids', 'Smoking', 'NoiseLevel',   'Music', 'Ambience'],
                       'business_payments':['AcceptsInsurance', 'BusinessAcceptsCreditCards','BusinessAcceptsBitcoin' ],
                       'accessibility':['WheelchairAccessible','ByAppointmentOnly','DriveThru','AgesAllowed', 'DogsAllowed']}
#groups
location = ['latitude', 'longitude']
working_hours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'is_open', 'Open24Hours']
parking = ['BikeParking','BusinessParking']
business_infra_features = ['OutdoorSeating', 'HasTV', 'WiFi']
business_extra_features = [ 'Alcohol','HappyHour','Caters', 'CoatCheck',  'GoodForKids', 'Smoking', 'NoiseLevel',   'Music', 'Ambience']
business_payments = ['AcceptsInsurance', 'BusinessAcceptsCreditCards','BusinessAcceptsBitcoin' ]
accessibility = ['WheelchairAccessible','ByAppointmentOnly','DriveThru','AgesAllowed', 'DogsAllowed']
no_reduction_factors = ['stars', 'review_count']

final_features_business = pd.DataFrame()
for i in groups_headers_dict:
    temp_all_features_business = all_features_business[groups_headers_dict[i]]
    temp_all_features_business.fillna(temp_all_features_business.mean(), inplace=True)
    factor_analysis_model = FactorAnalysis(n_components=1, random_state=0)
    modified_business_features = factor_analysis_model.fit_transform(temp_all_features_business)[:,0]
    final_features_business[i] = modified_business_features

for i in no_reduction_factors:
    final_features_business[i] = all_features_business[i]

final_features_business.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_factoranalysis.csv')