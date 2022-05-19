#Plantilla para el pre procesado de datos
#Importar el data set

dataset = read.csv('Data.csv')


#Dividir los datos en conjunto de entrenamiento 
install.packages("caTools")

library(caTools)
set.seed(123) #semilla aleatoria
#0.8 es el porcentaje que sive para entrenar (train)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

#para escalar los datos, se puede estandarizar o normalizar 
#Estandarizar permite aglutinar valores en torno a la media, da la campana de Gaus
# Normalización, transforma la data de 0 a 1, el valor más grnade tomará el valor de 1
#y el más pequeño en 0, se escala de forma lineal



##Escalado de valores (normalización)
### normalmente se hace, pero no siempre
# training_set [, 2:3] = scale(training_set[,2:3])
# testing_set [, 2:3]= scale(testing_set[, 2:3])













