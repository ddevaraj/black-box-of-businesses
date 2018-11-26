# Importing the libraries
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

# Importing the dataset
dataset = pd.read_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business_combined.csv')
X = dataset[['latitude','longitude','review_count','is_open','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Caters','WheelchairAccessible','BikeParking','AcceptsInsurance','BusinessAcceptsCreditCards','CoatCheck','HappyHour','GoodForKids','Open24Hours','OutdoorSeating','HasTV','BusinessAcceptsBitcoin','ByAppointmentOnly','DogsAllowed','DriveThru','Smoking','NoiseLevel','AgesAllowed','Alcohol','WiFi','Music','Ambience','BusinessParking','pos_count','neg_count','checkin_count']]
y = np.array(dataset[['stars']])

lab_enc = preprocessing.LabelEncoder()
label_dataset_encoded = lab_enc.fit_transform(y)

encoder = LabelBinarizer()
y = encoder.fit_transform(label_dataset_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

'''
#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
'''

def create_network():
    model = Sequential()
    model.add(Dense(5, input_shape=(37,), activation='relu'))
    model.add(Dense(9, activation='sigmoid'))
        
    #stochastic gradient descent
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

neural_network = create_network()
neural_network.fit(X,y, epochs=500, batch_size=10)