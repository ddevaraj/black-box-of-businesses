import pandas as pd
import numpy as np
import json

with open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset/yelp_academic_dataset_checkin.json") as inFIle:
    checkin_data = inFIle.readlines()
header = [x for x in json.loads(checkin_data[0]).keys()]
header = ['time', 'business_id']

checkin_final = []
for i in checkin_data:
    i = json.loads(i)
    checkin_final.append([i['time'], i['business_id']])

checkin_final = np.array(checkin_final)
checkin_csv = pd.DataFrame()
for i in range(len(header)):
    checkin_csv[header[i]] = checkin_final[:,i]

checkin_csv.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__checkin.csv')