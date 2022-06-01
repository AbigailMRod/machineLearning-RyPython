# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:07:18 2022

@author: Abigail
"""

#Regresión con árboles de decisión

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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#solo transform para que detecte la transformaci+on que se hizo en el de entrenamiento
X_test = sc_X.transform(X_test)"""


#Ajustar la regresión  con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state= 0)
regression.fit(X,y)

#predicción de nuestro modelo
y_pred =regression.predict(([[6.5]]))



#Visualización de los datos del modelo polinómico
X_grid = np.arange(min(X), max(X), 0.1) #para aumentar la cantidad de puntos qy no parazacan recatas las lineas de la curva
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color="red")
plt.plot(X,regression.predict((X)), color="blue")
plt.title("Modelo de regresión deárbol de decisión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en USD")