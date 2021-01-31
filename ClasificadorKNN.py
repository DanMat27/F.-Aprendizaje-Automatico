'''
Práctica 2
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from Clasificador import *
from Distancias import *
import numpy as np
from abc import ABCMeta,abstractmethod

class ClasificadorKNN(Clasificador):

  def __init__(self, k, distancia, norm):
    # Diccionario con las desviaciones y las medias de los atributos en entrenamiento
    self.dicc_desv_med = None
    # Tabla con datos normalizados en entrenamiento
    self.datos_normalizados = None
    # Valor K de vecinos
    self.k = k
    # Clase de la distancia que se va a utilizar
    self.distancia = distancia
    # Matriz de covarianzas inversa
    self.matriz_cov_inv = None
    # Variable que indica si se normalizan los datos
    self.norm = norm
    Clasificador.__init__(self)

  def calcularMediasDesv(self,datos,diccionario):
      # Calculamos la media y la desviacion estandar de cada atributo
      i = 0
      for atributo in diccionario:
        if(atributo != "Class"):
            self.dicc_desv_med[atributo] = {}

            # Calculo de la media de los valores del atributo
            sum = 0.0
            for dato in datos:
              sum += dato[i]
            self.dicc_desv_med[atributo]["Media"] = sum / len(datos)

            # Calculo de la desviacion estandar de los valores del atributo
            sum = 0.0
            for dato in datos:
              sum += (dato[i] - self.dicc_desv_med[atributo]["Media"])**2
            self.dicc_desv_med[atributo]["Desviacion"] = math.sqrt(sum / len(datos))

        i += 1

      #print(self.dicc_var_med)

  def normalizarDatos(self,datos,diccionario):
    # Normalizamos los datos y los almacenamos
    datos_normalizados = []
    for dato in datos:
        i = 0
        datoaux = []
        for atributo in diccionario:
            if(atributo != "Class"):
                norm = (dato[i] - self.dicc_desv_med[atributo]["Media"]) / self.dicc_desv_med[atributo]["Desviacion"]
                datoaux.append(norm)
            else:
                datoaux.append(dato[i])
            i += 1
        datos_normalizados.append(datoaux)

    #print(datos_normalizados)
    return datos_normalizados

  # Calcular medias y varianzas de los atributos
  # Normalizar los datos
  # Guardar en atributos del clasificador
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      self.datos_normalizados = None
      self.dicc_desv_med = {}

      self.calcularMediasDesv(datostrain,diccionario)

      if self.norm:
          self.datos_normalizados = self.normalizarDatos(datostrain,diccionario)
      else:
          self.datos_normalizados = datostrain

      # Calculamos la matriz de covarianzas inversa en caso de Distancia Mahalanobis
      if isinstance(self.distancia, DistanciaMahalanobis):
          self.matriz_cov_inv = None
          datos_sin_clase = []
          for dato in self.datos_normalizados:
              datos_sin_clase.append(dato[0:-1])
          matriz_cov = np.cov(np.transpose(datos_sin_clase))
          self.matriz_cov_inv = np.linalg.inv(matriz_cov)


  # Calcular la distancia que tiene el clasificador en todos los datos para el dato que se esta viendo
  # Obtenere sus k vecinos mas proximos
  # Realizar la clasificacion en funcion de la moda de la clase de esos vecinos
  def clasifica(self,datostest,atributosDiscretos,diccionario):

      # {"valor_distancia1":Clase, "valor_distancia2":Clase...}
      dicc_distancias = {}
      predicciones = []
      datostest_normalizados = None
      dicc_distancias_sort = None
      #tuple = (self.distancia.calculaDistancia(dato_cmp, dato), dato_cmp[-1])
      #distancias_clases.append(tuple)

      # Normalizamos los datos test
      if self.norm:
          datostest_normalizados = self.normalizarDatos(datostest, diccionario)
      else:
          datostest_normalizados = datostest

      # Calculamos distancias y le damos la clase moda de los K vecinos a dato
      for dato in datostest_normalizados:
          dicc_distancias = {}

          # Calculamos las distancias del dato Test a cada uno de los datos normalizados o no
          for dato_cmp in self.datos_normalizados:
              if isinstance(self.distancia, DistanciaMahalanobis):
                  dicc_distancias[self.distancia.calculaDistancia(dato_cmp, dato, self.matriz_cov_inv)] = dato_cmp[-1]
              else:
                  dicc_distancias[self.distancia.calculaDistancia(dato_cmp, dato)] = dato_cmp[-1]

          # Ordenamos distancias de menor a mayor
          dicc_distancias_sort = sorted(dicc_distancias.items())

          #print(dicc_distancias_sort)

          # Almacenamos las clases de los k vecinos mas cercanos
          vecinos = []
          i = 0
          for n in range(self.k):
              vecinos.append(dicc_distancias_sort[i][1])
              i += 1

          # Vemos que clase es la que mas se repite en los vecinos proximos
          clase1 = vecinos[0]
          c1 = 0
          c2 = 0
          for vecino in vecinos:
              if vecino == clase1:
                  c1 += 1
              else:
                  clase2 = vecino
                  c2 += 1

          # Clasificamos el dato Test en funcion de sus K vecinos
          if c1 < c2:
              predicciones.append(clase2)
          else:
              predicciones.append(clase1)

      return predicciones
