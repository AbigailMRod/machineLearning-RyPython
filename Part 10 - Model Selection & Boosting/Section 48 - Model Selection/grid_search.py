# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:31:09 2022

@author: Abigail
"""

#grid search 

#Cómo importar las librerías
import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



#Escalado de variables 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)


#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##############aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, 
                             X = X_train,
                             y = y_train,
                             cv = 10)
accuracies.mean()
accuracies.std()



###aplicar la mejora de grid search para optimizar el modelo y sus parámetros
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000],'kernel': ['linear']},
              {'C': [1, 10, 100, 1000],'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
              ]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

#la mejor presición
best_accuracy = grid_search.best_score_

#El mejor modelo
best_parameters = grid_search.best_params_
