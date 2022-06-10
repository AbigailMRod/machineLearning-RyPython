#Procesamiento del lenguaje natural

#importar el dataset 
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = F)


#Limpiar el dataset
#install.packages("tm")
library(tm)
corpus = VCorpus(VectorSource(dataset_original$Review))

#pasar a minusculas
corpus = tm_map(corpus, content_transformer(tolower))
#consultar el primer elemento del corpus ---> as.character(corpus[[1]])

#Eliminar los numero que hay dentro de las valoraciones
corpus = tm_map(corpus, removeNumbers)

#eliminar las puntuaciones
corpus = tm_map(corpus, removePunctuation)

#eliminar palabras irrelevantes 
#install.packages("SnowballC") para descargar las stopWords
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))

#convertir todas la palabras a su infinitivo (raíz)
corpus = tm_map(corpus, stemDocument)

#quitar espacios en blanco
corpus = tm_map(corpus, stripWhitespace)


#### matriz de caracteristicas. Crear el modelo de bag of words

dtm = DocumentTermMatrix(corpus)
#99.9% de palabras más frecuentes
dtm = removeSparseTerms(dtm, 0.999)

##
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

############random forest
#Codificar la variable de clasificación como factor
dataset$Liked = factor(dataset$Liked, 
                           levels = c(0,1))

#Dividir los datos en conjunto de entrenamiento 
install.packages("caTools")
library(caTools)
set.seed(123) #semilla aleatoria
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


#Ajustar el clasificador  con el conjunto de enrenamiento
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)


#Predicción con los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[, -692])
#type class es para que convierta la probabilidad solo en 2 factores

#Crear la matriz de confusión
cm = table(testing_set[,692], y_pred)


