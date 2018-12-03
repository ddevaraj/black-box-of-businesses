from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

def base_line_model(X_train, X_test, y_train, y_test):

    solver = ['lbfgs']
    results = {}
    final_coef = []
    final_result = []

    '''
    review_mean = np.array([review_mean]*len(test_label_set))
    results ['rmse'] = mean_squared_error(review_mean,test_label_set) ** 0.5
    '''

    for i in solver:
        clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 100000).fit(X_train,y_train)
        results['LogR'] = {}
        preds = clf.predict(X_test)
        results['LogR']['rmse'] = mean_squared_error(preds,y_test) ** 0.5
        results['LogR']['accuracy'] = clf.score(X_test,y_test)

        clf = RandomForestRegressor(max_depth=1000, random_state=0,n_estimators=100).fit(X_train,y_train)
        results['RanFR'] = {}
        preds = clf.predict(X_test)
        results['RanFR']['rmse'] = mean_squared_error(preds,y_test) ** 0.5
        results['RanFR']['accuracy'] = clf.score(X_test,y_test)

        clf = LinearRegression().fit(X_train,y_train)
        results['LinR'] = {}
        preds = clf.predict(X_test)
        results['LinR']['rmse'] = mean_squared_error(preds,y_test) ** 0.5
        results['LinR']['accuracy'] = clf.score(X_test,y_test)

        clf = MLPClassifier(solver='adam', learning_rate_init = 0.001,alpha=1e-5,hidden_layer_sizes=(15,10), random_state=1,max_iter=10000).fit(X_train,y_train)
        results['Neural'] = {}
        preds = clf.predict(X_test)
        results['Neural']['rmse'] = mean_squared_error(preds,y_test) ** 0.5
        results['Neural']['accuracy'] = clf.score(X_test,y_test)

    return results


def get_all_features(file_name):
    final_dataset = pd.read_csv(file_name)
    final_dataset = shuffle(final_dataset)#.sample(frac=1)

    final_dataset = final_dataset[['stars','checkin_count']]
    final_dataset = (final_dataset-final_dataset.mean())/final_dataset.std()
    final_dataset = final_dataset.sample(frac=1)
    final_dataset.fillna(final_dataset.mean(), inplace=True)
    label_dataset = final_dataset[['stars']]

    lab_enc = preprocessing.LabelEncoder()
    label_dataset_encoded = lab_enc.fit_transform(label_dataset)

    #baseline with review_count and linear regression
    feature_dataset = final_dataset[['checkin_count']]
    X_train, X_test, y_train, y_test = train_test_split(feature_dataset, label_dataset_encoded, test_size = 0.2)
    
    #baseline with average rating
    '''
    feature_dataset = final_dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking','pos_count','neg_count','checkin_count']]
    X_train, X_test, y_train, y_test = train_test_split(feature_dataset, label_dataset, test_size = 0.2)
    review_mean = y_train.median()
    print(review_mean)
    return y_train, y_test, review_mean
    '''

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_all_features('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_temp.csv')
results = base_line_model(X_train, X_test, y_train, y_test)
print(results)