# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:38:06 2022

@author: Abigail
"""

#Plantilla de Pre Procesado

#Cómo importar las librerías

import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Data.csv')
#iloc sirve para localizar los elemntos (filas y columnas) de un sataset
#por posicion i=index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


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


