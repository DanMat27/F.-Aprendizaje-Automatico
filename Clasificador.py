'''
Práctica 1
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import *


class Clasificador:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass


  @abstractmethod
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass


  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # Ademas obtiene la matriz de confusion y a partir de esta las tasas de confusion
  def error(self,datos,pred,diccionario):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    # Tasa error = Nerrores/Ntotal

    # Obtenemos los valores que puede tomar la clase, asignando el menor como
    # negativo y el mayor como positivo
    clases = diccionario['Class']

    i = 0
    for val in clases:
        if i == 0:
            N = clases[val]
            i += 1;
        else:
            P = clases[val]

    i = 0
    err = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # Obtenemos el numero de errores, falsos positivos, falsos negativos,
    # verdaderos positivos y verdaderos negativos
    for dato in datos:
        if dato[len(dato)-1] != pred[i]:
            err += 1
            if dato[len(dato)-1] == P and pred[i] == N:
                FN += 1
            if dato[len(dato)-1] == N and pred[i] == P:
                FP += 1
        else:
            if dato[len(dato)-1] == P and pred[i] == P:
                TP += 1
            if dato[len(dato)-1] == N and pred[i] == N:
                TN += 1
        i += 1

    #Guardamos los resultados en la matriz de matriz_confusion
    self.matriz_confusion = [TP, FP, FN, TN]

    #Calculamos las tasas correspondientes
    TPR = TP / (TP + FN)
    FNR = FN / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (FP + TN)
    self.tasas_confusion = [TPR, FNR, FPR, TNR]

    #Devolvemos la tasa de error
    return err/len(datos)


  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  def validacion(self,particionado,dataset,clasificador,seed=None):
    array_tasas_error = []

    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    array_particiones = particionado.creaParticiones(dataset.datos, seed)
    #print(len(array_particiones))

    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opcion es repetir la validacion simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
    self.matriz_confusion = []
    self.tasas_confusion = []
    for particion in array_particiones:
        clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain),dataset.nominalAtributos,dataset.diccionario)
        predicciones = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest),dataset.nominalAtributos,dataset.diccionario)
        error = clasificador.error(dataset.extraeDatos(particion.indicesTest),predicciones, dataset.diccionario)
        #print("La tasa de fallo ha sido " + str(error))
        array_tasas_error.append(error)

    return array_tasas_error

##############################################################################
