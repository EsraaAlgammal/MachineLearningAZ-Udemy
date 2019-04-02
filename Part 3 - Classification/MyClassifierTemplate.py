# Classifier Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classifier to the Training set
# Creating the regressor

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix # small letters: function not class
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Training set results
#for some reason this makes an error
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1), step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1), step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green')) (i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Explanation
from matplotlib.colors import ListedColormap
#create a copy of X_train and y_train
#for code resusability
X_set, y_set = X_train, y_train
#generate coordinate matrics using meshgrid
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
 np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
#Now we need to plot the prediction for every coordinate from X1 and X2
#therefore we need to first get the prediction from coordinate matrics
#convert X1 coordinate matrix to a flattened array - putting all the elements in one column
X1_ravel = X1.ravel()
#convert X2 coordinate matrix to a flattened array - putting all the elements in one column
X2_ravel = X2.ravel()
#create an array having 2 rows by placing X1_ravel over X2_ravel
X1X2_array = np.array([X1_ravel, X2_ravel])
#Since predict function takes an array which has 2 columns
#therefore we need to generate Transpose of X1X2_array - columns are converted into rows
X1X2_array_t = X1X2_array.T
#predict result using the classifier
X1X2_pred = classifier.predict(X1X2_array_t)
#result of prediction will be used to plot againt the coordinate matrics
#therefore we need to reshape the result to match the shape of coordinate matrics
#generated array contains prediction for every coordinate value
X1X2_pred_reshape = X1X2_pred.reshape(X1.shape)
#plot the predictions against coordinate matrics using contourf (filled)
result_plt = plt.contourf(X1, X2, X1X2_pred_reshape,
 alpha=0.75,
 cmap = ListedColormap(('red', 'green'))
)
#not mandatory
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#plot the actual points on the graph
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
 c = ListedColormap(('red', 'green'))(i), label = j)
#for housekeeping
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    


from matplotlib.colors import ListedColormap
#create a copy of X_test and y_test
#for code resusability
X_set, y_set = X_test, y_test
#generate coordinate matrics using meshgrid
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
 np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
#Now we need to plot the prediction for every coordinate from X1 and X2
#therefore we need to first get the prediction from coordinate matrics
#convert X1 coordinate matrix to a flattened array - putting all the elements in one column
X1_ravel = X1.ravel()
#convert X2 coordinate matrix to a flattened array - putting all the elements in one column
X2_ravel = X2.ravel()
#create an array having 2 rows by placing X1_ravel over X2_ravel
X1X2_array = np.array([X1_ravel, X2_ravel])
#Since predict function takes an array which has 2 columns
#therefore we need to generate Transpose of X1X2_array - columns are converted into rows
X1X2_array_t = X1X2_array.T
#predict result using the classifier
X1X2_pred = classifier.predict(X1X2_array_t)
#result of prediction will be used to plot againt the coordinate matrics
#therefore we need to reshape the result to match the shape of coordinate matrics
#generated array contains prediction for every coordinate value
X1X2_pred_reshape = X1X2_pred.reshape(X1.shape)
#plot the predictions against coordinate matrics using contourf (filled)
result_plt = plt.contourf(X1, X2, X1X2_pred_reshape,
 alpha=0.75,
 cmap = ListedColormap(('red', 'green'))
)
#not mandatory
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#plot the actual points on the graph
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
 c = ListedColormap(('red', 'green'))(i), label = j)
#for housekeeping
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()











