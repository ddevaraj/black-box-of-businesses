import pandas as pd
import numpy as np
import json

inFIle =  open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset/yelp_academic_dataset_review.json",'r')
outFile = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__review.csv','a')

review_data_header = inFIle.readline()
header = [x for x in json.loads(review_data_header).keys()]
header = ['review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool']

review_csv = pd.DataFrame()
for i in range(len(header)):
        review_csv[header[i]] = 0
review_csv.to_csv(outFile)
review_data_header = inFIle.readline()

while review_data_header is not None:
    
    review_data_header = json.loads(review_data_header)
    review_final = [review_data_header['review_id'], review_data_header['user_id'], review_data_header['business_id'], review_data_header['stars'], review_data_header['date'], review_data_header['text'], review_data_header['useful'], review_data_header['funny'], review_data_header['cool']]
    review_final = np.array(review_final)
    review_csv = pd.DataFrame()

    for i in range(len(header)):
        review_csv[header[i]] = [review_final[i]]

    review_csv.to_csv(outFile,header=False)
    review_data_header = inFIle.readline()

