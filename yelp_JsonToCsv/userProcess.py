import pandas as pd
import numpy as np
import json

inFIle =  open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset/yelp_academic_dataset_user.json",'r')
outFile = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__user.csv','a')

user_data_header = inFIle.readline()
header = [x for x in json.loads(user_data_header).keys()]
header = ['user_id', 'name', 'review_count', 'yelping_since', 'friends', 'useful', 'funny', 'cool', 'fans', 'elite', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos']


user_csv = pd.DataFrame()
for i in range(len(header)):
        user_csv[header[i]] = 0
user_csv.to_csv(outFile)
user_data_header = inFIle.readline()

while user_data_header is not None:
    
    user_data_header = json.loads(user_data_header)
    user_final = [user_data_header['user_id'], user_data_header['name'], user_data_header['review_count'], user_data_header['yelping_since'], user_data_header['friends'], user_data_header['useful'], user_data_header['funny'], user_data_header['cool'], user_data_header['fans'], user_data_header['elite'], user_data_header['average_stars'], user_data_header['compliment_hot'], user_data_header['compliment_more'], user_data_header['compliment_profile'], user_data_header['compliment_cute'], user_data_header['compliment_list'], user_data_header['compliment_note'], user_data_header['compliment_plain'], user_data_header['compliment_cool'], user_data_header['compliment_funny'], user_data_header['compliment_writer'], user_data_header['compliment_photos']]
    user_final = np.array(user_final)
    user_csv = pd.DataFrame()

    for i in range(len(header)):
        user_csv[header[i]] = [user_final[i]]

    user_csv.to_csv(outFile,header=False)
    user_data_header = inFIle.readline()