# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:46:04 2022

@author: Abigail 
"""

#Redes Neuronales Artificiales

#######################Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
#pip3 install Theano para Windows si no está instalado git 


#######################Instalar Tensorflow y Keras desde anaconda prompt
# conda install -c conda-forge keras
#pip3 install tensorflow

#////////////////////////////Parte 1, pre procesado de datos 
#Cómo importar las librerías
import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

###########################################################################
#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#trasnformar los paises en categorías
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])

#transformar el género en categoría
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough')

X = onehotencoder.fit_transform(X)
#para evitar multicolinealidad, se elimina una de las 3 columnas que se generaron 
# en las varibles dummy
X = X[:, 1:]

#########################################################################

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


##El escalado es obligatorio en redes neuronales  
#Escalado de variables 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


################################################################
#/////////////////// Parte 2 - Contruir la Red Neuronal Artificial

#Importar Keras y librerias adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

#Inicializar la red neuronal
classifier = Sequential()

#Añadir las capas de entrada y primer capa oculta de la red neuronal
#Dense es la conexion entre capas (la sinapsis)
#units es el número de nodos de la capa oculta, es aceptable utilizar la media entre
#los nodos de la capa de entrada y los nodos de la capa de salida
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu" , input_dim =11))
#relu (rectificador lineal unitario)
# input_dim es la dimension de entrada, en este caso 11 columnas 
#kernel_initializer para mantener lo pesos pequeños, cercanos a 0 pero no nulos

############# segunda capa oculta 
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu"))

############## capa de salida
classifier.add(Dense(units=1, kernel_initializer="uniform",
                     activation="sigmoid"))

#Compilar la red neuronal
#loss modelo que menos error tenga
classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=["accuracy"])


### ajustar la red neuronal al conjunto de entrenamiento
#batch_size, procesar "n" elmentos y corregir los pesos
#epochs es el numero deiteraciones que se harán
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


#//////////////////// Parte 3 Evaluar el modelo y calcular predicciones finales
#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)
#umbral de abandono, para convertir de probabilidad a categoría
y_pred =(y_pred>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

