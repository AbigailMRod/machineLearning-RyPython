# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:11:06 2022

@author: Abigail
"""

#Clustering jerarquico

#Importar las librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importar los datos
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values


#utilizar el dendrograma para encontrar el numero optimo de cluster
#Se intenta minimizar la varianza que hay entre los puntos del cluster con ward
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")


#ajustar el clustering jerarquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X) 

#visualización de los clusters 
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=100, c="red", label="Cautos")
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=100, c="blue", label="Estándar")
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=100, c="green", label="Objetivo")
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=100, c="cyan", label="Descuidados")
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=100, c="magenta", label="Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles USD)")
plt.ylabel("Puntuación de gastos (1-100)")
plt.legend()






