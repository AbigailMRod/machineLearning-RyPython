# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:04:36 2022

@author: Abigail
"""

#SVR (máquinas de soporte para regresión)

#Cómo importar las librerías

import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
#iloc sirve para localizar los elemntos (filas y columnas) de un sataset
#por posicion i=index
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
#size=0.2 implica que un 20% de los datos se destinaran a testing, max0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_trai, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Escalado de variables 
#Se hace a menudo pero no siempre
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))



#Ajustar la regresión  con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)


#predicción de nuestro modelo con SVR
y_pred =(regression.predict(sc_X.transform(np.array([[6.5]]))))
y_pred_inverse = sc_y.inverse_transform(regression.predict( sc_X.transform(np.array([[6.5]]))))



#Visualización de los datos del modelo SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")


