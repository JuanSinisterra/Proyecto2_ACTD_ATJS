################################################################################
################################################################################
############################# TAREA 4 - MODELAMIENTO ###########################
##################################### ACTD #####################################
################################################################################



Datos.Modelamiento=read.csv("~/9no Semestre/Analítica Computacional para la Toma de Decisiones/Proyecto 2/Tarea 4 - Modelamiento/Datos Modelamiento.csv", stringsAsFactors=TRUE)

columnas_a_convertir=c("X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11", "X24", "X25", 
                       "X26", "X27", "X28", "X29")

Datos.Modelamiento[columnas_a_convertir]=lapply(Datos.Modelamiento[columnas_a_convertir], 
                                                as.character)


modelo=glm(Y ~ ., data = Datos.Modelamiento, family = "binomial")
summary(modelo)

library(car)

vif(modelo)

