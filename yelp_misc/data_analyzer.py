import pandas as pd


business_1 = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_temp.csv')
states = business_1['state']
print(states)


'''
unique_states = business_1['state'].unique()
print(len(unique_states))
print(unique_states)

states = {}
for i in unique_states:
    states[i] = len(states)

print(states)
'''