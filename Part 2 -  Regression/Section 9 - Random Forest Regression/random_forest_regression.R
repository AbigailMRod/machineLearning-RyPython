#regresión con árboles aleatorios


#Importar el data set

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

#Dividir los datos en conjunto de entrenamiento 
#install.packages("caTools")

# library(caTools)
# set.seed(123) #semilla aleatoria
# #0.8 es el porcentaje que sive para entrenar (train)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)

#para escalar los datos, se puede estandarizar o normalizar 
#Estandarizar permite aglutinar valores en torno a la media, da la campana de Gaus
# Normalizaci?n, transforma la data de 0 a 1, el valor m?s grnade tomar? el valor de 1
#y el m?s peque?o en 0, se escala de forma lineal



##Escalado de valores (normalizaci?n)
### normalmente se hace, pero no siempre
# training_set [, 2:3] = scale(training_set[,2:3])
# testing_set [, 2:3]= scale(testing_set[, 2:3])


#Ajustar el modelo de regresión de árboles aleatorios
install.packages("randomForest")
library(randomForest)
set.seed(1234)
regression = randomForest(x= dataset[1], 
                          y= dataset$Salary,
                          ntree = 500)


#predicción de nuevos resultados con regresión polinómica
y_pred= predict(regression, newdata = data.frame(Level = 6.5))



#Visualización del modelo 

library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot()+ geom_point(aes(x= dataset$Level, y= dataset$Salary),
                     color= "red") +
  geom_line(aes(x = x_grid, y= predict(regression, newdata = data.frame(Level=x_grid))), 
            color= "blue") +
  ggtitle("Modelo de árboles aleatorios") +
  xlab("Nivel del empleado") +
  ylab("Sueldo en USD")
