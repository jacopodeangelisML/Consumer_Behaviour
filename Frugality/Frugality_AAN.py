# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:06:41 2020

@author: Iacopo
"""
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#Load data
df = pd.read_csv('data.csv', sep = ',')
df.info()
df['FRUGALITY'].value_counts()

#Check null values
null = df.isnull().sum()

#Dicotomize outcome variable and create new outcome column
df.loc[df['FRUGALITY'] >4, 'frugality'] = 1
df.loc[df['FRUGALITY'] <= 4, 'frugality'] = 0

#Delete old outcome column
df = df.drop('FRUGALITY', axis = 1)
df = df.iloc[:,3:217]

##RESAMPLE THE OUTCOME
#Import SMOTE library  
from imblearn.over_sampling import SMOTE
#Resample the minority class
sm = SMOTE(sampling_strategy='minority', random_state=7)
#Fit the model to generate the data, and create X and Y
X, y = sm.fit_sample(df.drop('frugality', axis=1), df['frugality'])

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

##ARTIFICIAL NEURAL NETWORK
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initialising the ANN
model = Sequential()
#Adding the input layer and the first hidden layer
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 213))
#Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#Compiling the ANN
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 50)
#Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
#Evaluation
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(roc_auc_score(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Hyperparameter optimization
def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 213))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model.fit(X_train, y_train, batch_size = 100, epochs = 500)

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

k_model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
random_search = RandomizedSearchCV(estimator=k_model, 
                                   param_distributions=param, 
								   cv= 3, 
								   verbose=0,
                                   scoring='accuracy')
random_search.fit(X_train, y_train)

print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
params = random_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
