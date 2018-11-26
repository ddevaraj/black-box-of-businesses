import pandas as pd
import numpy as np
import json

with open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset/yelp_academic_dataset_tip.json") as inFIle:
    tip_data = inFIle.readlines()
header = [x for x in json.loads(tip_data[0]).keys()]
header = ['text', 'date', 'likes', 'business_id', 'user_id']

tip_final = []
for i in tip_data:
    i = json.loads(i)
    tip_final.append([i['text'], i['date'], i['likes'], i['business_id'], i['user_id']])

tip_final = np.array(tip_final)
tip_csv = pd.DataFrame()
for i in range(len(header)):
    tip_csv[header[i]] = tip_final[:,i]

tip_csv.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__tip.csv')