import pandas as pd
import numpy as np
import json

with open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset/yelp_academic_dataset_business.json") as inFIle:
    business_data = inFIle.readlines()

#header = [x for x in json.loads(business_data[0]).keys()]

header = ['business_id', 'name', 'neighborhood', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories']# removed 'attributes' and 'hours'
working_hours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] # added 'hours'
attribute_keys_completeBinary = ['Caters','WheelchairAccessible','BikeParking', 'AcceptsInsurance', 'BusinessAcceptsCreditCards','CoatCheck', 'HappyHour', 'GoodForKids', 'Open24Hours', 'OutdoorSeating', 'HasTV', 'BusinessAcceptsBitcoin','ByAppointmentOnly', 'DogsAllowed', 'DriveThru']
attribute_keys_completeMulti = ['Smoking', 'NoiseLevel', 'AgesAllowed', 'Alcohol', 'WiFi']
multi_key_mapping = {'Smoking': {'no':0, 'yes':1,'outdoor':2}, 'NoiseLevel' : {'quiet':0,'average':1,'loud':2, 'very_loud':3},
                    'WiFi': {'no':0,'free':1,'paid':2}, 'AgesAllowed' : {'allages':0,'18plus':1,'19plus':2,'21plus':3},
                    'Alcohol' : {'none':0, 'beer_and_wine' : 1, 'full_bar':2}}
attribute_keys_completeJson = ['Music','Ambience','BusinessParking'] # removed 'HairSpecializesIn','BestNights','DietaryRestrictions','GoodForMeal'
json_key_mapping = {'Music': {'no_music':0, 'dj':1, 'background_music':2,'karaoke':3, 'live':4, 'video':5, 'jukebox':6},
                    'Ambience': {'romantic':1, 'intimate':2, 'classy':3, 'hipster':4, 'touristy':5, 'trendy':6, 'upscale':7, 'casual':8, 'divey':9},
                    'BusinessParking': {'garage': 1, 'street': 2, 'validated': 3, 'lot': 4, 'valet': 5}}
#'HairSpecializesIn': ['coloring', 'africanamerican', 'curly', 'perms', 'kids', 'extensions', 'asian', 'straightperms'], 
#'BestNights': ['monday', 'tuesday', 'friday', 'wednesday', 'thursday', 'sunday', 'saturday'], 
#'DietaryRestrictions': ['dairy-free', 'gluten-free', 'vegan', 'kosher', 'halal', 'soy-free', 'vegetarian'], 
#'GoodForMeal': ['dessert', 'latenight', 'lunch', 'dinner', 'breakfast', 'brunch']}
attribute_keys_uncommon = ['RestaurantsGoodForGroups','HairSpecializesIn','BestNights','GoodForDancing','RestaurantsTakeOut','DietaryRestrictions','RestaurantsPriceRange2','BYOBCorkage','Corkage','BYOB', 'RestaurantsDelivery','RestaurantsCounterService','GoodForMeal','RestaurantsReservations', 'RestaurantsAttire', 'RestaurantsTableService' ]
final_header = []

'''
for i in business_data:
    i = json.loads(i)
    for j in attribute_keys_completeJson:
        if i['attributes'] is not None and i['attributes'].get(j) is not None:
            i['attributes'][j] = i['attributes'][j].replace("'",'"')
            i['attributes'][j] = i['attributes'][j].replace("False",'0')
            i['attributes'][j] = i['attributes'][j].replace("True",'1')
            print(i['attributes'][j])
            k = json.loads(i['attributes'][j])
            for l in k.keys():
                if l not in completeJson[j]:
                    completeJson[j].append(l)
print(completeJson)
'''

business_final = []
business_final_attributes = []

for i in business_data:
    i = json.loads(i)

    temp_bussiness = []

    for k in header:
        temp_bussiness.append(i[k])
    
    if i['hours'] != '' and i['hours'] is not None:
        for j in working_hours:
            if i['hours'].get(j) is not None:
                temp_bussiness.append('1')
            else:
                temp_bussiness.append('0')
    else:
        for j in working_hours:
            temp_bussiness.append('0')

    for k in attribute_keys_completeBinary:
        if i['attributes'] is None:
            i['attributes'] = {}
            temp_bussiness.append('0')
        elif k not in i['attributes']:
            temp_bussiness.append('0')
        else:
            if i['attributes'][k] == 'True':
                temp_bussiness.append('1')
            elif i['attributes'][k] == 'False':
                temp_bussiness.append('0')
            else:
                temp_bussiness.append(i['attributes'][k])

    for k in attribute_keys_completeMulti:
        if k not in i['attributes']:
            temp_bussiness.append('0')
        else:
            temp_bussiness.append(str(multi_key_mapping[k][i['attributes'][k]]))


    for k in attribute_keys_completeJson:
        if k not in i['attributes']:
            temp_bussiness.append('0')
        else:
            i['attributes'][k] = i['attributes'][k].replace("'",'"')
            i['attributes'][k] = i['attributes'][k].replace("False",'0')
            i['attributes'][k] = i['attributes'][k].replace("True",'1')
            i['attributes'][k] = json.loads(i['attributes'][k])
            tempKVal = '0'
            for j in i['attributes'][k]:
                if i['attributes'][k][j] == 1:
                    tempKVal = str(json_key_mapping[k][j])
                    break
            temp_bussiness.append(tempKVal)

    '''
    if i['attributes'] is not None and i['attributes'].get(j) is not None:
    i['attributes'][j] = i['attributes'][j].replace("'",'"')
    i['attributes'][j] = i['attributes'][j].replace("False",'0')
    i['attributes'][j] = i['attributes'][j].replace("True",'1')
    print(i['attributes'][j])
    k = json.loads(i['attributes'][j])
    for l in k.keys():
        if l not in completeJson[j]:
            completeJson[j].append(l)


    for j in attribute_keys_completeJson:
        if j not in i['attributes']:
            i['attributes'][j] = '0'
    for j in working_hours:
        i[j] = '0'

    temp_bussiness = []
    for k in header:
        temp_bussiness.append(i[k])
    temp_bussiness_attributes = []
    for k in attribute_keys_complete:
        if k == "BusinessParking" and i['attributes'][k] != 0 :
            i['attributes'][k] = i['attributes'][k].replace("'",'"')
            i['attributes'][k] = i['attributes'][k].replace("False",'0')
            i['attributes'][k] = i['attributes'][k].replace("True",'1')
            i['attributes'][k] = json.loads(i['attributes'][k])
            for l in i['attributes'][k]:
                if i['attributes'][k][l] == '1':
                    i['attributes'][k] = 1
                    temp_bussiness.append(i['attributes'][k])
                    break
            i['attributes'][k] = 0
            temp_bussiness_attributes.append(i['attributes'][k])
        else:
            temp_bussiness_attributes.append(i['attributes'][k])

    '''

    business_final.append(temp_bussiness)
    #business_final_attributes.append(temp_bussiness_attributes)
    #business_final.append([i['business_id'], i['name'], i['neighborhood'], i['address'], i['city'], i['state'], i['postal_code'], i['latitude'], i['longitude'], i['stars'], i['review_count'], i['is_open'], i['attributes'], i['categories'], i['hours']])

business_final = np.array(business_final)
all_headers = []
all_headers.extend(header)
all_headers.extend(working_hours)
all_headers.extend(attribute_keys_completeBinary)
all_headers.extend(attribute_keys_completeMulti)
all_headers.extend(attribute_keys_completeJson)
print(all_headers)
business_csv = pd.DataFrame()
for i in range(len(all_headers)):
    business_csv[all_headers[i]] = business_final[:,i]
business_csv.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business.csv')