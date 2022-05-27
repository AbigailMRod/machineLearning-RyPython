# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:36:24 2022

@author: Abigail
"""

#Regresión polinómica

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

"""No se hará  división de prueba y entrenamiento porque son muy pocos los
datos que se tienen y sería perjudicial quitar un punto del modelo"""

#Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Ajustar la regresión polinpomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
#primer término es el grado (se puede ir cambiando el numero para ajustar el modelo) 
poly_reg = PolynomialFeatures(degree=4)
X_poly =poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#Visualización de los datos del modelo linela
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Modelo de regresión lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en USD")


#Visualización de los datos del modelo polinómico
X_grid = np.arange(min(X), max(X), 0.1) #para aumentar la cantidad de puntos qy no parazacan recatas las lineas de la curva
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en USD")

#predicción de nuestro modelo
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))





