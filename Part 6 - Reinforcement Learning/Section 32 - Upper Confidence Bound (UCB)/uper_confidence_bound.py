# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:16:36 2022

@author: Abigail 
"""

#upper Confidence Bound (UCB)

#importar las librerias 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cargar el dataset 
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

##Algoritmo de upper conficende bound
import math
N = 10000 #número de rondas (usuarios)
d = 10 #numero de anuncion que salen
number_of_selections = [0] * d #cuantas veces ha sido seleccionado cada anuncio
sum_of_rewards = [0] * d #suma de recompensas
ads_selected = []
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if (number_of_selections[i]>0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

##visualización
#histograma
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("Id del anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")

