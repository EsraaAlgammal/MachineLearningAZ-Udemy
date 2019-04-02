# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #avoiding the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Part 2 - Now Let's make the ANN!!!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential(optimizer = 'adam')

# Adding the input layer and the first hidden layer

# 6 = 11 input layers + 1 output layer (cause the output is either 0 or 1, 
# so it's just one output layer) /2 = 12/2 = 6 which is the average number of Hidden Layers here!!!
# uniform initialized the weights randomly, and make sure they have small numbers close to 0
# relu in the rectifier function - used for the hidden layers. We'll use Sigmoid for the output layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) 

# Adding a second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) # we dont need the input dim, cause this is the 2nd layer - it was needed in the 1st for obvious reasons :D

# Adding the output layer
#note: 'sigmoid' is used when output_dim is 1, 'softmax' is a sigmoid rectifier used when output_dim >1 (or: dependent variable that has 1 or more categories)
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# adam is an efficient stochastic gradient "decent" 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # setting a shreshold - should be higher for more sensitive problems like in the medical field

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix # small letters: function not class
cm = confusion_matrix(y_test, y_pred)
#accuracy = (1544 + 140) /2000 = 0.842 = 84%
