import pandas as pd


business_1 = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business.csv')
business_2 = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/ExtraBusinessReviewAttributes.csv')
business_3 = pd.merge(business_1,business_2,on='business_id').to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_combined.csv')
