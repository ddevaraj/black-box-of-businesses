import pandas as pd
import numpy as np
import json

with open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset/yelp_academic_dataset_photo.json") as inFIle:
    photo_data = inFIle.readlines()
header = [x for x in json.loads(photo_data[0]).keys()]
header = ['photo_id', 'business_id', 'caption', 'label']

photo_final = []
for i in photo_data:
    i = json.loads(i)
    photo_final.append([i['photo_id'], i['business_id'], i['caption'], i['label']])

photo_final = np.array(photo_final)
photo_csv = pd.DataFrame()
for i in range(len(header)):
    photo_csv[header[i]] = photo_final[:,i]

photo_csv.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__photo.csv')