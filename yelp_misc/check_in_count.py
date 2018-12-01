import csv
import numpy as np
import json
import ast

business_dict = {}
with open('yelp_checkin.csv','r') as file:
    data = csv.DictReader(file)
    for row in data:
        b_id = row['business_id']
        check_dict = ast.literal_eval(row['time'])
        if b_id not in business_dict:
            business_dict[b_id] = 0
        count = 0
        for val in check_dict.values():
            count += val
        business_dict[b_id] = count

sorted_d = sorted((-value, key) for (key, value) in business_dict.items())
print(sorted_d[0])

with open('check_in_count.json','w') as json_file:
    json.dump(business_dict, json_file)

with open('check_in_count.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in business_dict.items():
       writer.writerow([key, str(value)])


