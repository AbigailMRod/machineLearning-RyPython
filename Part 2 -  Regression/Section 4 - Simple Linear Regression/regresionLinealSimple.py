# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:12:22 2022

@author: Abigail
"""

#Regresión lineal simple

#Cómo importar las librerías

import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Salary_Data.csv')
#iloc sirve para localizar los elemntos (filas y columnas) de un sataset
#por posicion i=index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Dividir el data set en conjunto d eentrenamiento y conjunto de test
#size=0.2 implica que un 20% de los datos se destinaran a testing, max0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)



#Escalado de variables 
#Se hace a menudo pero no siempre
#Para el caso de la regresión lineal simple no se necesita escalar 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#solo transform para que detecte la transformaci+on que se hizo en el de entrenamiento
X_test = sc_X.transform(X_test)"""


#Crear el modelo de regresión lineal simple con el conjunto de ENTRENAMIENTO
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de TEST 
y_pred = regression.predict(X_test)

## Visualizar los resultados de ENTRENAMIENTO
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Suelvo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (US)")

## Visualizar los resultados de TEST
#la recta es la misma para ambos grupos de datos, lo que cambia son los puntos
plt.scatter(X_test, y_test, color="green")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Suelvo vs Años de experiencia (Conjunto de test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (US)")









