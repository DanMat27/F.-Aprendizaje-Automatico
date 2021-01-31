'''
Práctica 1
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from abc import ABCMeta,abstractmethod
import random


class Particion():

  # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor
  def __init__(self):
      self.particiones = [] #Array de particiones del particionado

  @abstractmethod
  def creaParticiones(self,datos,seed=None):
    pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    def __init__(self, proporcionTest, numeroEjecuciones):
        self.proporcionTest = proporcionTest       #Proporcion para parte de test
        self.numeroEjecuciones = numeroEjecuciones #Numero de particiones generadas
        EstrategiaParticionado.__init__(self)

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el numero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        self.particiones = [] #Reseteamos el array de particiones para crear las nuevas

        # Calculamos el numero de datos de cada parte (segun proporcion)
        id = int(self.proporcionTest*len(datos))

        # Usamos semilla si hay
        if seed is not None:
            random.seed(seed)

        # Bucle de creacion de las particiones (segun numeroEjecuciones)
        n = 1
        while n <= self.numeroEjecuciones:
            p = Particion()

            # Lista de indices de Test de manera aleatoria
            cont = 0
            while cont < id:
                indice = random.randint(0, len(datos)-1)

                if indice not in p.indicesTest:
                    p.indicesTest.append(indice)
                    cont += 1

            # Lista de indices de Train, que son los que no están en Test
            list_indices = []
            i = 0
            while i < len(datos):
                if i not in p.indicesTest:
                    list_indices.append(i)
                i += 1

            # Desordenamos la lista ordenada anterior para evitar sesgo
            random.shuffle(list_indices)
            p.indicesTrain = list_indices

            # Aniadimos la particion con las listas anteriores a la lista de particiones
            self.particiones.append(p)
            n += 1

        return self.particiones

#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, numeroParticiones):
        self.numeroParticiones = numeroParticiones #Numero de particiones generadas
        EstrategiaParticionado.__init__(self)

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        self.particiones = [] #Reseteamos el array de particiones para crear las nuevas
        
        # Calculamos el numero de datos de cada parte (segun numeroParticiones)
        num_datos_particion = int(len(datos)/self.numeroParticiones)

        # Usamos semilla si hay
        if seed is not None:
            random.seed(seed)

        # Creamos los diferentes K (numeroParticiones) conjuntos
        array_conjuntos = []
        indices_usados = []
        n = 1
        while n <= self.numeroParticiones:
            cont = 0
            list_aux = []
            # Listas de indices de manera aleatoria
            while cont < num_datos_particion:
                indice = random.randint(0, len(datos)-1)

                if indice not in indices_usados:
                    list_aux.append(indice)
                    indices_usados.append(indice)
                    cont += 1

            array_conjuntos.append(list_aux)
            n += 1

        #Bucle de creacion de las diferentes particiones
        n = 1
        while n <= self.numeroParticiones:
            p = Particion()

            # El Test cada vez es un conjunto distinto en cada particion
            p.indicesTest = array_conjuntos[n-1]

            i = 0
            list_aux = []
            # El resto de conjuntos forman el Train
            while i < self.numeroParticiones:
                if i != n-1:
                    list_aux = list_aux + array_conjuntos[i]

                i += 1

            p.indicesTrain = list_aux

            # Aniadimos la particion con las listas anteriores a la lista de particiones
            self.particiones.append(p)
            n += 1

        return self.particiones
