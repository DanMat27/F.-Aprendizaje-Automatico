'''
Práctica 1
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from Clasificador import *
from abc import ABCMeta,abstractmethod
import numpy as np
import math
import scipy as sp

class ClasificadorNaiveBayes(Clasificador):

  def __init__(self, laplace):
      self.laplace = laplace     #Indica si se aplica Laplace
      self.verosimilitudes = {}  #{'A1': {'1':{'du':x,'bl':y,'no':z},'2':{'du':x,'bl':y,'no':z}}...}
      self.prioris = {}          #{'negative':x, 'positive':y}
      self.matriz_confusion = [] #Matriz de confusion de este clasificador
      self.tasas_confusion = [] #Tasas de conjusion de este clasificador
      Clasificador.__init__(self)


  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
      numdatos = len(datosTrain)
      self.prioris = {}
      self.verosimilitudes = {}
      aux_clases = {}

      #Calculamos las probabilidades a priori
      for clase in diccionario['Class']:
          self.prioris[clase] = 0
          for dato in datosTrain:
              if dato[len(atributosDiscretos)-1] == diccionario['Class'][clase]:
                 self.prioris[clase] += 1
          aux_clases[clase] = self.prioris[clase]
          self.prioris[clase] /= numdatos

      #print(self.prioris)


      i = 0
      flag_laplace = 0
      #Creamos la tabla correspondiente a cada atributo
      for atributo in diccionario:
          #Si es discreto (== Nominal)
          if atributosDiscretos[i] == True and atributo != 'Class':
              self.verosimilitudes[atributo] = {}
              #Contamos para cada valor del atributo cuantos hay de cada clase {'b':{'+':x,'-':y}, 'o':{'+':x,'-':y}, 'x':{'+':x,'-':y}...}
              for valor in diccionario[atributo]:
                  self.verosimilitudes[atributo][valor] = {}
                  for clase in diccionario['Class']:
                      self.verosimilitudes[atributo][valor][clase] = 0
                      for tupla in datosTrain:
                          if tupla[i] == diccionario[atributo][valor]:
                              if tupla[len(atributosDiscretos)-1] == diccionario['Class'][clase]:
                                  self.verosimilitudes[atributo][valor][clase] += 1
                      #Si hay alguna probabilidad 0, se aplica despues el +1 de Laplace si este es True
                      if self.verosimilitudes[atributo][valor][clase] == 0 and self.laplace == True:
                          flag_laplace = 1
                      else:
                          self.verosimilitudes[atributo][valor][clase] /= aux_clases[clase]

              #Si se encontro algun 0 antes, se repiten los calculos sumando 1 (Laplace)
              if flag_laplace == 1:
                  cont_laplace = 0
                  for valor in diccionario[atributo]:
                      cont_laplace += 1
                      self.verosimilitudes[atributo][valor] = {}
                      for clase in diccionario['Class']:
                          #Sumamos 1 al numerador
                          self.verosimilitudes[atributo][valor][clase] = 1
                          for tupla in datosTrain:
                              if tupla[i] == diccionario[atributo][valor]:
                                  if tupla[len(atributosDiscretos)-1] == diccionario['Class'][clase]:
                                      self.verosimilitudes[atributo][valor][clase] += 1
                  for valor in diccionario[atributo]:
                      for clase in diccionario['Class']:
                          #Sumamos el numero de valores que puede tomar el atributo al denominador
                          self.verosimilitudes[atributo][valor][clase] /= (aux_clases[clase] + cont_laplace)

          #Si es continuo (!= Nominal)
          elif atributosDiscretos[i] == False and atributo != 'Class':
              self.verosimilitudes[atributo] = {}
              #Calculamos la media y la varianza de cada clase {'Media':{'+':x,'-':y},'Varianza':{'+':x,'-':y}}}
              self.verosimilitudes[atributo]['Media'] = {}
              self.verosimilitudes[atributo]['Varianza'] = {}
              for clase in diccionario['Class']:
                  self.verosimilitudes[atributo]['Media'][clase] = 0
                  self.verosimilitudes[atributo]['Varianza'][clase] = 0
                  aux_media_cuadrados = 0
                  #Calculamos la Media
                  for tupla in datosTrain:
                      if tupla[len(atributosDiscretos)-1] == diccionario['Class'][clase]:
                          self.verosimilitudes[atributo]['Media'][clase] += tupla[i]
                  self.verosimilitudes[atributo]['Media'][clase] /= aux_clases[clase]
                  #Calculamos la Varianza var[x] = sum1_N (x_i - Media(x))^2 / N
                  for tupla in datosTrain:
                      if tupla[len(atributosDiscretos)-1] == diccionario['Class'][clase]:
                          aux_resta = tupla[i] - self.verosimilitudes[atributo]['Media'][clase]
                          aux_media_cuadrados += aux_resta*aux_resta
                  aux_media_cuadrados /= aux_clases[clase]
                  self.verosimilitudes[atributo]['Varianza'][clase] = aux_media_cuadrados


          i += 1

      #print(self.verosimilitudes)


  def clasifica(self,datostest,atributosDiscretos,diccionario):
      prob_posteriori = [] #[{P('+'|D):x},P('-'|D):y},...]
      #P(clase|dato) = multiplicacion verosimilitudes_datos * priori_clase ( / priori_datos normalizando )
      #P(Class=+|a1=x,a2=y,...,a7=z) = P(a1=x,a2=y,...,a7=z|Class=+)*P(Class=+) / P(a1=x,a2=y,...,a7=z) =
      #= P(a1=x|Class=+)*...*P(a7=z|Class=+)*P(Class=+) / p
      predicciones = []

      d = -1
      for dato in datostest:
          d += 1
          prob_posteriori.append({})
          #Obtenemos el valor real
          for clase in diccionario['Class']:
              prob_posteriori[d][clase] = 1
              i = 0
              for atributo in diccionario:
                  #Si es discreto (== Nominal)
                  if atributosDiscretos[i] == True and atributo != 'Class':
                      for valor in diccionario[atributo]:
                          if diccionario[atributo][valor] == dato[i]:
                              valor_real = valor
                      #Multiplicamos las verosimilitudes de atributos independientes
                      prob_posteriori[d][clase] *= self.verosimilitudes[atributo][valor_real][clase]

                  #Si es continuo (!= Nominal)
                  #N(x|Media,Varianza) = [1/(2*pi*Varianza)^(1/2)] * e^[(-1/(2*Varianza))*(x - Media)^2]
                  elif atributosDiscretos[i] == False and atributo != 'Class':
                      ''' La funcion de Dist.Normal de Scipy es lenta
                      sigma = self.verosimilitudes[atributo]['Varianza'][clase]
                      mu = self.verosimilitudes[atributo]['Media'][clase]
                      x = dato[i]
                      prob_posteriori[d][clase] *= sp.stats.norm(mu,sigma).pdf(x)
                      '''
                      #Calculo del primer termino que multiplica
                      f = (1 / ((2*math.pi*self.verosimilitudes[atributo]['Varianza'][clase])**(1/2)))
                      #Calculo del segundo termino que multiplica
                      s = (math.exp( (-1 / (2*self.verosimilitudes[atributo]['Varianza'][clase])) * ((dato[i] - self.verosimilitudes[atributo]['Media'][clase])**2)))
                      #Calculo final multiplicando ambos terminos
                      prob_posteriori[d][clase] *= (f * s)

                  i += 1
              #Multiplicamos por la priori de la clase
              prob_posteriori[d][clase] *= self.prioris[clase]

      #Decision final MAP
      for prob in prob_posteriori:
          flag = 0
          for post in prob:
              if flag < prob[post]:
                  flag = prob[post]
                  prediccion = post
          predicciones.append(diccionario['Class'][prediccion])

      #print(predicciones)

      #Devolvemos array numpy con las predicciones
      return np.array(predicciones)
