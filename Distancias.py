'''
Práctica 2
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from abc import ABCMeta,abstractmethod
import math
import numpy as np
from scipy.spatial import distance

# Clase abstracta de una Distancia
class Distancia:

  # Clase abstracta
  __metaclass__ = ABCMeta

  @abstractmethod
  def calculaDistancia(self,x,test):
    pass

########################################################################

# Distancia Euclidea
class DistanciaEuclidea(Distancia):

    def __init__(self):
        Distancia.__init__(self)

    def calculaDistancia(self,x,test):
        i = 0
        sum = 0.0
        while i < len(x)-1:
            sum += (x[i] - test[i])**2
            i += 1

        return math.sqrt(sum)

########################################################################

# Distancia Manhattan
class DistanciaManhattan(Distancia):

    def __init__(self):
        Distancia.__init__(self)

    def calculaDistancia(self,x,test):
        i = 0
        sum = 0.0
        while i < len(x)-1:
            res = x[i] - test[i]
            if res < 0:
                res *= -1
            sum += res
            i += 1

        return sum

########################################################################

# Distancia Mahalanobis
class DistanciaMahalanobis(Distancia):

    def __init__(self):
        Distancia.__init__(self)

    def calculaDistancia(self,x,test,matriz_cov_inv):

        val = distance.mahalanobis(x[0:-1], test[0:-1], matriz_cov_inv)

        return val

########################################################################
