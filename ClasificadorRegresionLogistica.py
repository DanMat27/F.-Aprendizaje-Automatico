'''
Práctica 2
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from Clasificador import *
import numpy as np
from abc import ABCMeta,abstractmethod
import random
import math

class ClasificadorRegresionLogistica(Clasificador):

 def __init__(self, nep, η, norm):
     # Diccionario con las desviaciones y las medias de los atributos en entrenamiento
     self.dicc_desv_med = {}
     # Tabla con datos normalizados en entrenamiento
     self.datos_normalizados = []
     # Numero de epocas
     self.n_epocas = nep
     # Constante de aprendizaje
     self.constante_aprendizaje = η
     # Vector w / Frontera de decision
     self.w = None
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

 # Calcula del valor de la funcion sigmoidal del numero pasado
 def funcion_sigmoidal(self, num):
     try:
         return 1 / (1 + math.exp(-num))
     except:
        return 0

 # Calcula el producto escalar entre dos vectores
 def producto_escalar(self, vector1, vector2):
     resultado = 0.0

     i = 0
     while(i < len(vector2)):
         resultado += vector1[i]*vector2[i]
         i += 1

     return resultado

 def multiplicar_valor_vector(self, vector, valor):
     vector_m = []
     i = 0

     while(i < len(vector)):
         vector_m.append(vector[i]*valor)
         i += 1

     return vector_m

 def resta_vectores(self, vector1, vector2):
     i = 0
     vector_r = []

     while(i < len(vector1)):
         vector_r.append(vector1[i] - vector2[i])
         i += 1

     return vector_r

 def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      # Normalizamos los datos
      self.calcularMediasDesv(datostrain,diccionario)
      if self.norm:
          self.datos_normalizados = self.normalizarDatos(datostrain,diccionario)
      else:
          self.datos_normalizados = datostrain

      # Generamos w inicial aleatoriamente w = {w0, w1, ..., wD} D numero atributos
      i = 0
      self.w = []
      while(i < len(atributosDiscretos)):
        self.w.append(random.uniform(-0.5, 0.5))
        i += 1

      i = 0
      # Vamos actualizando el valor de los coeficientes de w con los datos de Train en nEpocas
      while(i < self.n_epocas):
          for dato in self.datos_normalizados:
              # Aniadimos x0 al dato
              dato_x0 = [1]
              for d in dato[:-1]:
                  dato_x0.append(d)
              #dato_x0 = [1] + dato[:-1]
              # Producto escalar de w y el dato
              prod_escalar = self.producto_escalar(self.w, dato_x0)
              # Valor de la funcion sigmoidal que equivale a P(C1|x)
              sigma = self.funcion_sigmoidal(prod_escalar)
              # Actualizamos w --> w - η*(sigma-t)*x
              aux = self.constante_aprendizaje * (sigma - dato[-1])
              m = self.multiplicar_valor_vector(dato_x0, aux);
              self.w = self.resta_vectores(self.w, m)

          i += 1
      #print(self.w)


 def clasifica(self,datostest,atributosDiscretos,diccionario):
     predicciones = []

     # Normalizamos los datostest
     if self.norm:
         datos = self.normalizarDatos(datostest,diccionario)
     else:
         datos = datostest

     for dato in datos:
         dato_x0 = [1] + dato[:-1] # Aniadimos x0 al dato
         # Si el producto escalar entre w y dato_x0 < 0 clase 2, si no clase 2
         if self.producto_escalar(self.w, dato_x0) < 0:
             predicciones.append(0)
         else:
             predicciones.append(1)

     return predicciones
