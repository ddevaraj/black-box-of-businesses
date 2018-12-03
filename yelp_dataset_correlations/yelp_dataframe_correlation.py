import numpy as np
import pandas as pd

headers = ['business_id', 'name', 'neighborhood', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Caters', 'WheelchairAccessible', 'BikeParking', 'AcceptsInsurance', 'BusinessAcceptsCreditCards', 'CoatCheck', 'HappyHour', 'GoodForKids', 'Open24Hours', 'OutdoorSeating', 'HasTV', 'BusinessAcceptsBitcoin', 'ByAppointmentOnly', 'DogsAllowed', 'DriveThru', 'Smoking', 'NoiseLevel', 'AgesAllowed', 'Alcohol', 'WiFi', 'Music', 'Ambience', 'BusinessParking']
final_headers = ['latitude', 'longitude', 'review_count', 'is_open', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Caters', 'WheelchairAccessible', 'BikeParking', 'AcceptsInsurance', 'BusinessAcceptsCreditCards', 'CoatCheck', 'HappyHour', 'GoodForKids', 'Open24Hours', 'OutdoorSeating', 'HasTV', 'BusinessAcceptsBitcoin', 'ByAppointmentOnly', 'DogsAllowed', 'DriveThru', 'Smoking', 'NoiseLevel', 'AgesAllowed', 'Alcohol', 'WiFi', 'Music', 'Ambience', 'BusinessParking']
label = 'stars'
all_features_business_original = pd.read_csv("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business.csv")
all_features_business = all_features_business_original[final_headers]
rows, cols = all_features_business.shape

flds = list(all_features_business.columns)

corr = all_features_business.corr().values

feature_correlations_all = set()
for i in range(cols-1):
    for j in range(i+1, cols-1):
        feature_correlations_all.add((flds[i],flds[j],corr[i,j]))

feature_correlations_all_out = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_correlations/yelp_feature_correlations.txt','w')
for i in feature_correlations_all:
    feature_correlations_all_out.write(str(i)+"\n")

feature_correlations_threshold = set()
for i in range(cols-1):
    for j in range(i+1, cols-1):
        if np.abs(corr[i,j]) > 0.2:
            feature_correlations_threshold.add((flds[i],flds[j],corr[i,j]))

feature_correlations_threshold_out = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_correlations/yelp_feature_correlations_threshold.txt','w')
for i in feature_correlations_threshold:
    feature_correlations_threshold_out.write(str(i)+"\n")

label_correlations = set()
for i in final_headers:
    corr_features = all_features_business_original[[i,'stars']]
    label_correlations.add((i,'stars',corr_features.corr().values[0][1]))

label_correlations_out = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_correlations/yelp_label_correlations.txt','w')
for i in label_correlations:
    label_correlations_out.write(str(i)+"\n")

print(label_correlations)
print("**********")
print(feature_correlations_threshold)
print("**********")
print(feature_correlations_all)