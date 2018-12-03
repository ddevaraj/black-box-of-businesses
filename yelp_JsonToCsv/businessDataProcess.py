import pandas as pd
import numpy as np
import json

with open("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset/yelp_academic_dataset_business.json") as inFIle:
    business_data = inFIle.readlines()

#header = [x for x in json.loads(business_data[0]).keys()]
location_kmeans_headers = ['location15','location20']
location_kmeans_15 = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_location_15.csv')
location_kmeans_15 = location_kmeans_15.drop(['Unnamed: 0'], axis=1)
location_kmeans_20 = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_location_20.csv')
location_kmeans_20 = location_kmeans_20.drop(['Unnamed: 0'], axis=1)
state = {'AB': 0, 'NV': 1, 'QC': 2, 'AZ': 3, 'ON': 4, 'PA': 5, 'OH': 6, 'IL': 7, 'WI': 8, 'NC': 9, 'BY': 10, 'NYK': 11, 'SC': 12, 'C': 13, 'XGM': 14, 'ST': 15, 'IN': 16, 'RP': 17, 'CMA': 18, 'NI': 19, 'NLK': 20, 'VS': 21, '6': 22, 'CO': 23, 'HE': 24, 'VA': 25, 'RCC': 26, '01': 27, 'SG': 28, 'NY': 29, 'OR': 30, 'NW': 31, '4': 32, '10': 33, 'CC': 34, 'CA': 35, '45': 36, 'LU': 37, 'MT': 38, 'G': 39, 'PO': 40, 'B': 41, 'VT': 42, 'AL': 43, 'WAR': 44, 'MO': 45, 'HU': 46, 'M': 47, 'AR': 48, 'O': 49, 'FL': 50, 'WA': 51, 'KY': 52, 'CRF': 53, 'TAM': 54, 'NE': 55, 'XMS': 56, 'GA': 57, 'AG': 58, 'WHT': 59, 'MA': 60, 'V': 61, 'BC': 62, 'SP': 63, 'DE': 64, 'HH': 65, '11': 66, 'CS': 67, 'MN': 68}
cityOut = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/city_names.json','r')
cityOut = json.load(cityOut)
#headers_notused = ['name', 'neighborhood', 'address', 'postal_code']
header = ['business_id','latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories']# removed 'attributes' and 'hours'
city_header = ['city']
working_hours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']# added 'hours'
working_hours_rep_2 = ['Weekday','Weekend']
attribute_keys_completeBinary = ['Caters','WheelchairAccessible','RestaurantsTakeOut','BikeParking', 'AcceptsInsurance', 'BusinessAcceptsCreditCards','CoatCheck', 'HappyHour', 'GoodForKids', 'Open24Hours', 'OutdoorSeating', 'HasTV', 'BusinessAcceptsBitcoin', 'DogsAllowed', 'DriveThru']
attributes_nomod = ['RestaurantsPriceRange2']
attribute_keys_completeMulti = ['Smoking', 'NoiseLevel', 'AgesAllowed', 'Alcohol', 'WiFi']
multi_key_mapping = {'Smoking': {'no':0, 'yes':1,'outdoor':2}, 'NoiseLevel' : {'quiet':0,'average':1,'loud':2, 'very_loud':3},
                    'WiFi': {'no':0,'free':1,'paid':2}, 'AgesAllowed' : {'allages':0,'18plus':1,'19plus':2,'21plus':3},
                    'Alcohol' : {'none':0, 'beer_and_wine' : 1, 'full_bar':2},
                    'state' : {'AB': 0, 'NV': 1, 'QC': 2, 'AZ': 3, 'ON': 4, 'PA': 5, 'OH': 6, 'IL': 7, 'WI': 8, 'NC': 9, 'BY': 10, 'NYK': 11, 'SC': 12, 'C': 13, 'XGM': 14, 'ST': 15, 'IN': 16, 'RP': 17, 'CMA': 18, 'NI': 19, 'NLK': 20, 'VS': 21, '6': 22, 'CO': 23, 'HE': 24, 'VA': 25, 'RCC': 26, '01': 27, 'SG': 28, 'NY': 29, 'OR': 30, 'NW': 31, '4': 32, '10': 33, 'CC': 34, 'CA': 35, '45': 36, 'LU': 37, 'MT': 38, 'G': 39, 'PO': 40, 'B': 41, 'VT': 42, 'AL': 43, 'WAR': 44, 'MO': 45, 'HU': 46, 'M': 47, 'AR': 48, 'O': 49, 'FL': 50, 'WA': 51, 'KY': 52, 'CRF': 53, 'TAM': 54, 'NE': 55, 'XMS': 56, 'GA': 57, 'AG': 58, 'WHT': 59, 'MA': 60, 'V': 61, 'BC': 62, 'SP': 63, 'DE': 64, 'HH': 65, '11': 66, 'CS': 67, 'MN': 68}}
attribute_keys_completeJson = ['Music','Ambience','BusinessParking'] # removed 'HairSpecializesIn','BestNights','DietaryRestrictions','GoodForMeal'
json_key_mapping = {'Music': {'no_music':0, 'dj':1, 'background_music':2,'karaoke':3, 'live':4, 'video':5, 'jukebox':6},
                    'Ambience': {'romantic':1, 'intimate':2, 'classy':3, 'hipster':4, 'touristy':5, 'trendy':6, 'upscale':7, 'casual':8, 'divey':9},
                    'BusinessParking': {'garage': 1, 'street': 2, 'validated': 3, 'lot': 4, 'valet': 5}}
parking = ['BusinessParking']
#'HairSpecializesIn': ['coloring', 'africanamerican', 'curly', 'perms', 'kids', 'extensions', 'asian', 'straightperms'], 
#'BestNights': ['monday', 'tuesday', 'friday', 'wednesday', 'thursday', 'sunday', 'saturday'], 
#'DietaryRestrictions': ['dairy-free', 'gluten-free', 'vegan', 'kosher', 'halal', 'soy-free', 'vegetarian'], 
#'GoodForMeal': ['dessert', 'latenight', 'lunch', 'dinner', 'breakfast', 'brunch']}
attribute_keys_uncommon = ['RestaurantsGoodForGroups','HairSpecializesIn','BestNights','GoodForDancing','DietaryRestrictions','BYOBCorkage','Corkage','BYOB', 'RestaurantsDelivery','RestaurantsCounterService','GoodForMeal','RestaurantsReservations', 'RestaurantsAttire', 'RestaurantsTableService' ]
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

    for k in city_header:
        if i[k] == '':
            temp_bussiness.append('0')
        else:
            temp_bussiness.append(cityOut[i[k]])
    
    if i['hours'] != '' and i['hours'] is not None:
        for j in working_hours:
            if i['hours'].get(j) is not None:
                temp_bussiness.append('1')
            else:
                temp_bussiness.append('0')
    else:
        for j in working_hours:
            temp_bussiness.append('0')

    for k in working_hours_rep_2:
        if k == 'Weekday':

            if temp_bussiness[7] == '1' or temp_bussiness[8] == '1' or temp_bussiness[9] == '1' or temp_bussiness[10] == '1' or temp_bussiness[11] == '1':
            #if i['hours']['Monday'] == '1' or i['hours']['Tuesday'] == '1' or i['hours']['Wednesday'] == '1' or i['hours']['Thursday'] == '1' or i['hours']['Friday'] == '1':
                temp_bussiness.append('1')
            else:
                temp_bussiness.append('0')
        
        if k == 'Weekend':
            if temp_bussiness[12] == '1' or temp_bussiness[13] == '1':
            #if i['hours']['Saturday'] == '1' or i['hours']['Sunday'] == '1':
                temp_bussiness.append('1')
            else:
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

    for k in attributes_nomod:
        if k not in i['attributes']:
            temp_bussiness.append('0')
        else:
            temp_bussiness.append(str(i['attributes'][k]))

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

    for k in parking:
        if k not in i['attributes']:
            temp_bussiness.append('0')
        else:
            tempKVal = 0
            for j in i['attributes'][k]:
                if i['attributes'][k][j] == 1:
                    tempKVal += 1
            temp_bussiness.append(str(tempKVal))

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
all_headers.extend(city_header)
all_headers.extend(working_hours)
all_headers.extend(working_hours_rep_2)
all_headers.extend(attribute_keys_completeBinary)
all_headers.extend(attributes_nomod)
all_headers.extend(attribute_keys_completeMulti)
all_headers.extend(attribute_keys_completeJson)
all_headers.extend(parking)
business_csv = pd.DataFrame()
for i in range(len(all_headers)):
    business_csv[all_headers[i]] = business_final[:,i]

business_csv = pd.merge(business_csv,location_kmeans_15,on='business_id')
business_csv = pd.merge(business_csv,location_kmeans_20,on='business_id')

bi_uni_gram = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/unibi200.csv')
business_csv = pd.merge(business_csv,bi_uni_gram,on='business_id')

tip_reviews = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/tip_features.csv')
business_csv = pd.merge(business_csv,tip_reviews,on='business_id')

user_reviews = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/features.csv')
user_reviews = user_reviews[['business_id','pos_count','neg_count','checkin_count', 'text_length', 'count_punctuation']]
business_csv = pd.merge(business_csv,user_reviews,on='business_id')

business_csv.drop('Unnamed: 0_x', axis=1)
business_csv.drop('Unnamed: 0_y', axis=1)

business_csv['total_pos_count'] = business_csv['pos_count_x'] + business_csv['pos_count_y']
business_csv['total_neg_count'] = business_csv['neg_count_x'] + business_csv['neg_count_y']
business_csv['total_punctuation_count'] = business_csv['count_punctuation_x'] + business_csv['count_punctuation_y']
business_csv['total_text_length'] = (business_csv['text_length_x'] + business_csv['text_length_y'])/2


business_csv.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_temp.csv')