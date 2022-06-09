#Sistema de recomendación, regla de apriori

#Preprocesado de datos
install.packages("arules")
library(arules)
dataset =read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv", 
                            sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=20) #topN es el numero de items que queremos ver

##Entrenar el algoritmo apriori con el dataset
rules = apriori(data = dataset, 
                parameter = list(support = 0.004, confidence= 0.2))

#saber el soporte de manera dinámica  
# venta de al menos 3 vecesa al dia 
#la venta es por semana (7 días )
#7500 son los items vendidos en la semana 
##### reglas de asociacion creadas con este nivel de soporte 

############# el nivel de confianza va a depender de los objetivos que persigue 
#cada empresa, si se selecciona un nivel muy alto habra reglas obvias, 
#si se selecciona un nivvel muy pequeño habrá reglas que ocurren muy rara vez (casualidades)
#el valor recomendado es 0.8




#Ordenasr las reglas de asociacion de la más fuerte a la más debil
inspect(sort(rules, by='lift')[1:10])

install.packages("arulesViz")
library(arulesViz)
plot(rules, method = "graph", engine = "htmlwidget")
