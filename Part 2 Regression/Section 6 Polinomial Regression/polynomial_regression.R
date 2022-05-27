#Regresión polinómica

#Importar el data set

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]


#Dividir los datos en conjunto de entrenamiento 
install.packages("caTools")

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


#Ajustar modelo re regresión lineal cone l conjunto de datos
lin_reg = lm(formula = Salary ~ . ,
             data= dataset)
summary(lin_reg)

#Ajustar el modelo de regresion polinomica con el conjunto de datos
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ . ,
              data= dataset)
summary(poly_reg)

#Visualización del modelo lineal
library(ggplot2)
ggplot()+ geom_point(aes(x= dataset$Level, y= dataset$Salary),
                     color= "red") +
          geom_line(aes(x = dataset$Level, y= predict(lin_reg, newdata = dataset)), 
                    color= "blue") +
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo en USD")

#Visualizaciónl del modelo polnómico
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot()+ geom_point(aes(x= dataset$Level, y= dataset$Salary),
                     color= "red") +
  geom_line(aes(x = x_grid, y= predict(poly_reg, newdata = data.frame(Level=x_grid,
                                                                      Level2 = x_grid^2,
                                                                      Level3 = x_grid^3,
                                                                      Level4 = x_grid^4))), 
            color= "blue") +
  ggtitle("Predicción polinómica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo en USD")

# predicción de nuevos resultados con regresión lineal
y_pred= predict(lin_reg, newdata = data.frame(Level = 6.5))

#predicción de nuevos resultados con regresión polinómica
y_pred_poly= predict(poly_reg, newdata = data.frame(Level = 6.5,
                                               Level2 = 6.5^2,
                                               Level3 = 6.5^3,
                                               Level4 = 6.5^4))

