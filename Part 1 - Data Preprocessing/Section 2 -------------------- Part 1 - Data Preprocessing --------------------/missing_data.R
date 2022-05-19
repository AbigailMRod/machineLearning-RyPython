#Plantilla para el pre procesado de datos - Datos faltantes
#Importar el data set

dataset = read.csv('Data.csv')

#Tratamiento de los valores NA (se sustituyen por medias)
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = T)),
                     dataset$Age)


dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = T)),
                        dataset$Salary)