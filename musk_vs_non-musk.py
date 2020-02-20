# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:42:38 2020

@author: Payal Arora
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the csv file
data = pd.read_csv('musk_csv.csv')
data.head()

#checking for null values
data.isnull().sum()

"""checking the correlation among features and dropping features with high correlation"""

# Creating correlation matrix
corr_matrix = data.corr().abs()

# Selecting upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
print(upper)

# Finding index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)
len(to_drop)

# Drop features 
df=data.drop(data[to_drop], axis=1)
df.head()
X = df.iloc[:, 3:-1]
y = df.iloc[:,-1]

#splitting the dataset into 80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Normalising the features
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

"""Applying ANN"""

import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.python.keras.layers

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'he_uniform',activation='relu',input_dim = 134))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)

print(model_history.history.keys())

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Making predictions and evaluating the model"""
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculating the Accuracy,precision,recall and f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

accuracy=accuracy_score(y_pred,y_test)
print(accuracy)
precision=precision_score(y_pred,y_test)
print(precision)
recall=recall_score(y_pred,y_test)
print(recall)
f1=f1_score(y_pred,y_test)
print(f1)
