# -*- coding: utf-8 -*-
"""Diabetes Prediction Model.ipynb

Original file is located at
    https://colab.research.google.com/drive/1uKYiSRlQNlDie8Xrx0ELrQORmrEWO4-Z

# ***Importing the Dependencies***
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""# ***Data collection and analysis***
PIMA Diabetes Dataset.
It only contain data from females

"""

# loading the dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

#printing the first 5 rows of the dataset
diabetes_dataset.head()

#number of rows and coloums in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0--> Non-Diabetic                             

1-->Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

print(x)

print(X)

print(Y)

"""# ***Data standardization***"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""# ***Train Test Split***"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""# Training the model"""

classifier = svm.SVC(kernel='linear')

# Training the support vector machine classifier
classifier.fit(X_train, Y_train)

"""# ***Model Evaluation***

Accuracy Score
"""

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the training data : ', test_data_accuracy)

"""# ***Making a predictive system ***"""

#Enter your datasets in the form of the in the diabetes file.
input_data = (5,166,72,19,175,25.8,0.587,51)


# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data) 


# reshape the array as we are predicting for one instance
reshaped_data = input_data_as_numpy_array.reshape(1,-1)


# standardize the input data
standardize_input_data = scaler.transform(reshaped_data)
print(standardize_input_data)


prediction = classifier.predict(standardize_input_data)


print(prediction)

if(prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

