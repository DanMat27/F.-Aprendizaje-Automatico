'''
Práctica 3
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
from Clasificador import *
import math
import random
import copy
import numpy as np
from abc import ABCMeta,abstractmethod
from sklearn.preprocessing import OneHotEncoder

class ClasificadorAlgoritmoGenetico(Clasificador):


  def __init__(self, num_generaciones, tam_poblacion, max_reglas, tipo_cruce, prob_cruce, prob_mutacion):
    Clasificador.__init__(self)
    self.num_generaciones = num_generaciones
    self.tam_poblacion = tam_poblacion
    self.max_reglas = max_reglas
    self.tipo_cruce = tipo_cruce # 1 = Uniforme y 2 = En un punto
    self.prob_cruce = prob_cruce
    self.prob_mutacion = prob_mutacion
    #self.elitismo = elitismo
    self.fitness_medios_gens = []
    self.fitness_best_gens = []

  # Genera un individuo de la poblacion
  def genera_individuo_aleatorio(self):
    # Semilla
    random.seed() 
    ind = []
    # Numero de reglas que tendra el individuo [1, max_reglas] 
    num_reglas = random.randint(1, self.max_reglas)
    i = 0
    # Bucle de generacion de reglas aleatorias
    while(i < num_reglas): #ind[r1[10101010...],r2[01010...]...]
      regla = []
      for j in range(0, self.tam_ind):
          regla.append(random.randint(0, 1))

      c1 = 0
      c0 = 0
      # Miramos si la regla tiene todo 1s o 0s
      for r in regla:
        if(r == 1):
          c1 += 1
        else:
          c0 += 1
          
      # Si es asi, se ignora
      if(c0 == self.tam_ind or c1 == self.tam_ind):
        i -= 1
      # Si no, se aniade
      else:
        ind.append(regla)
        
      i += 1

    return ind

  
  # Genera la poblacion inicial de individuos
  def genera_poblacion_inicial(self,diccionario):
    # Calculamos tamanio de una regla en funcion de los atributos
    self.tam_ind = 0
    for atributo in diccionario:
      self.tam_ind += len(diccionario[atributo])
    self.tam_ind -= 1
    
    # Generamos los tam_poblacion individuos de la poblacion inicial
    pob = []
    i = 0
    while(i < self.tam_poblacion):
      ind = self.genera_individuo_aleatorio()
      pob.append(ind)
      i += 1

    return pob

      
  # Obtenemos la cantidad de bits que representan cada atributo
  # Generamos la población inicial de forma aleatoria (cadenas de bits)
  # Convertir datostrain a binario con el mismo formato
  # Bucle de generaciones realizando el fitness, cruce y mutacion de los individuos
  # Seleccion de la nueva generacion (con elitismo)
  # Al final, se obtiene el mejor individuo
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    if(self.matriz_confusion == []):
      self.fitness_medios_gens = []
      self.fitness_best_gens = []
    
    # Generamos la población inicial de forma aleatoria
    self.poblacion = self.genera_poblacion_inicial(diccionario)
    
    # Convertimos los datostrain a binario con el mismo formato
    datostrain_sinclase = []
    for dato in datostrain:
      datostrain_sinclase.append(dato[0:len(dato)-1])
    enc = OneHotEncoder()
    enc.fit(datostrain_sinclase)
    datosTrain_binarios = enc.transform(datostrain_sinclase).toarray()
    
    self.datosTrain = []
    for dato in datosTrain_binarios:
      self.datosTrain.append(dato.tolist())

    i = 0
    for d in self.datosTrain:
      d.append(datostrain[i][-1])
      i += 1
    
    # Semilla
    random.seed()

    # Arrays medias y mejores fitness
    self.medias_fitness = []
    self.mejores_fitness = []

    # Bucle de generaciones realizando el fitness, cruce y mutacion de los individuos
    for gen in range(self.num_generaciones):  
 
      # Calculo del fitness de la poblacion
      # Fitness del individuo i = % de aciertos sobre los datos de entrenamiento
      fitness = []
      for individuo in self.poblacion:
        
        num_aciertos = 0
        for dato in self.datosTrain:
          # Hallamos la clase que predice el individuo sobre ese dato
          # Sino cumple ninguna regla devuelve -1 (error)
          c = self.claseEnFuncionReglas(individuo, dato, diccionario)

          # Si no hay error
          if(c != -1):
            # Comprobamos si es acierto
            if (c == dato[-1]):
              num_aciertos += 1

        # Guardamos su fitness en un array 
        fitness.append(num_aciertos)  

      #print("Fitness generacion: ")
      #print(fitness)

      # Elitismo 
      # Cogemos los dos mejores individuos, es decir, con mayor fitness
      # Calculamos el numero de mejores individuos tenemos que guardar
      n_elitistas = (int)(0.05*self.tam_poblacion)
      
      array_elitistas = []
      mejores_fitness = [] 
      individuos_mejor = []
      for k in range(n_elitistas):
        mejores_fitness.append(-1)
        individuos_mejor.append(-1)

      for k in range(n_elitistas):
        i = 0
        for f in fitness:
          if(f > mejores_fitness[k] and i not in individuos_mejor): 
            individuos_mejor[k] = i
            mejores_fitness[k] = f
          i += 1
      
      # Los guardamos aparte en array_elitistas
      for k in range(n_elitistas):
        #print("F" + str(k) + " " + str(mejores_fitness[k]))
        #print("i" + str(k) + " " + str(individuos_mejor[k]))
        array_elitistas.append(copy.deepcopy(self.poblacion[individuos_mejor[k]]))
    
      #print(array_elitistas)

      # Realizamos la seleccion de progenitores proporcional a fitness  
      S = self.ruleta_fitness(fitness)

      # Cruce
      S = self.cruce(S)

      # Mutacion
      S = self.mutacion(S, self.datosTrain, diccionario)

      # Seleccion de supervivientes de la generacion
      self.poblacion = S

      # Volvemos a meter los elitistas guardados 
      # Calculamos de nuevo el fitness para saber que dos individuos eliminar
      fitness = []
      for individuo in self.poblacion:
        num_aciertos = 0
        for dato in self.datosTrain:
          # Hallamos la clase que predice el individuo sobre ese dato NO PAIN NO GAIN
          # Sino cumple ninguna regla devuelve -1 (error)
          c = self.claseEnFuncionReglas(individuo, dato, diccionario)

          # Si no hay error
          if(c != -1):
            # Comprobamos si es acierto
            if (c == dato[-1]):
              num_aciertos += 1

        # Guardamos su fitness en un array 
        fitness.append(num_aciertos)

      #print("Fitness tras cruce: ")
      #print(fitness)

      # Protegemos a los dos individuos con mejor fitness y a los dos individuos con peor fitness supervivientes
      mejores_fitness = [-1,-1]
      peores_fitness = [1000,1000]
      individuos_mejor = [0,0]
      individuos_peor = [0,0]

      i = 0
      # Mejores
      for f in fitness:
        if(f > mejores_fitness[0]): 
          individuos_mejor[0] = i
          mejores_fitness[0] = f
        i += 1
      i = 0
      for f in fitness:
        if(f > mejores_fitness[1] and i != individuos_mejor[0]):
          individuos_mejor[1] = i
          mejores_fitness[1] = f
        i += 1
        
      i = 0
      # Peores
      for f in fitness:
        if(f < peores_fitness[0]): 
          individuos_peor[0] = i
          peores_fitness[0] = f
        i += 1
      i = 0
      for f in fitness:
        if(f < peores_fitness[1] and i != individuos_peor[0]):
          individuos_peor[1] = i
          peores_fitness[1] = f
        i += 1

      # Se meten los elitistas 
      cont = 0
      naux = []
      while(True):
        n = random.randint(0, self.tam_poblacion-1)
        if(n != individuos_mejor[0] and n != individuos_mejor[1]):
          if(n != individuos_peor[0] and n != individuos_peor[1]):
            if(n not in naux):
              self.poblacion[n] = array_elitistas[cont].copy()
              #print("n " + str(n))
              naux.append(n)
              cont += 1
        
        if(cont >= n_elitistas): 
          break
      
      # Almacenamos el mejor fitness de cada generacion
      # Tambien se almacena la media del fitness de la generacion
      mejor_fitness = -1
      mejor_individuo = -1
      fitness_total = 0.0
      cont = 0
      for individuo in self.poblacion:
        
        num_aciertos = 0
        for dato in self.datosTrain:
          # Hallamos la clase que predice el individuo sobre ese dato
          # Sino cumple ninguna regla devuelve -1 (error)
          c = self.claseEnFuncionReglas(individuo, dato, diccionario)

          # Si no hay error
          if(c != -1):
            # Comprobamos si es acierto
            if (c == dato[-1]):
              num_aciertos += 1
        
        # Mejor individuo
        if(num_aciertos > mejor_fitness):
          mejor_fitness = num_aciertos
          mejor_individuo = cont

        # Vamos almacenando los fitness para la media
        fitness_total += num_aciertos 
        cont += 1
      
      # Calculamos la media y almacenamos ambos valores
      fitness_total /= self.tam_poblacion
      self.medias_fitness.append(fitness_total)
      self.mejores_fitness.append(mejor_fitness)

      # print(self.poblacion)
    
    # Poblacion final
    # print(self.poblacion)

    # Nos quedamos con el mejor individuo para la clasificación
    self.mejor_individuo = self.poblacion[mejor_individuo].copy()
    print("Fitness del mejor individuo: " + str(mejor_fitness))
    #print(self.mejor_individuo)

    self.fitness_medios_gens.append(self.medias_fitness.copy())
    self.fitness_best_gens.append(self.mejores_fitness.copy())


  # Realiza la mutacion de los progenitores
  def mutacion(self, progenitores,dato, diccionario):
        
    array_mutacion = []

    # Separamos los progenitores en dos, los que mutan y los que no
    index = []
    for i in range(len(progenitores)):
      n = random.random()
      if(n < self.prob_mutacion):
        array_mutacion.append(progenitores[i])
    
    for a in array_mutacion:
      progenitores.remove(a)

    #print(array_mutacion)

    # Realizamos una mutacion al azar por cada individuo
    for prog in array_mutacion:
      
      n = random.randint(0, 2)

      # Control de los individuos problematicos
      if(len(prog) == 1): 
        n = 0
      elif (len(prog) == self.max_reglas): 
        n = 1
     
      # Vamos a aniadir una regla al individuo
      if(n == 0):
        # Generamos una nueva regla de forma aleatoria
        regla = []
        for j in range(0, self.tam_ind):
          regla.append(random.randint(0, 1))
        
        prog.append(regla) 

      # Vamos a quitar una regla al individuo
      elif(n == 1):
        # Elegimos la regla a eliminar de forma aleatoria
        n = random.randint(0, len(prog)-1)

        prog.remove(prog[n])
      
      # Vamos a hacer bit-flip
      else:
        # Elegimos la regla que vamos a modificar
        n1 = random.randint(0, len(prog)-1)
        # Elegimos el bit de la regla
        n2 = random.randint(0, self.tam_ind-1) 
        if(prog[n1][n2] == 1): 
          prog[n1][n2] = 0
        else:
          prog[n1][n2] = 1   

    '''print(array_mutacion)
    print("Fitness de los dos primeros individuos mutados")
    num_aciertos = 0
    num_aciertosl = 0
    for dato in self.datosTrain:
      c = self.claseEnFuncionReglas(array_mutacion[1], dato, diccionario)
      l = self.claseEnFuncionReglas(array_mutacion[0], dato, diccionario)
      # Si no hay error
      if(c != -1):
        if (c == dato[-1]):
          num_aciertos += 1
      # Si no hay error
      if(l != -1):
        if (l == dato[-1]):
          num_aciertosl += 1

    print(num_aciertos)
    print(num_aciertosl)'''

    # Se vuelven a meter en la poblacion
    for prog in array_mutacion:
      progenitores.append(prog)

    # print(len(progenitores))
        
    return progenitores


  # Realiza el cruce de los progenitores
  def cruce(self, progenitores):
        
    array_cruces = []
    
    # Separamos los progenitores en dos, los que sufren el cruce y los que no
    index = []
    for i in range(len(progenitores)):
      n = random.random()
      if(n < self.prob_cruce):
        array_cruces.append(progenitores[i])
    
    for a in array_cruces:
      progenitores.remove(a)
    
    # Realizamos el cruce en pares
    # Si los progenitores que se van a cruzar son impares ignoramos el ultimo
    if(len(array_cruces) % 2 != 0):
      progenitores.append(array_cruces[-1])
      array_cruces = array_cruces[:-1]
    
    #print(array_cruces)
    #print(progenitores)
    #print(len(array_cruces))
    #print(len(progenitores))
     
    i = 0
    while(i < len(array_cruces)):
      # Cruzamos dos individuos
      # Cogemos una regla de cada individuo de manera aleatoria 
      n1 = random.randint(0, len(array_cruces[i])-1)
      n2 = random.randint(0, len(array_cruces[i+1])-1)
      hijo1 = []
      hijo2 = []
      
      # Usamos un tipo de cruce indicado por argumento
      if(self.tipo_cruce == 1):
        self.cruce_uniforme(n1, n2, array_cruces, hijo1, hijo2, i)
      else:
        self.cruce_un_punto(n1, n2, array_cruces, hijo1, hijo2, i)

      # Modificamos las reglas cruzadas
      array_cruces[i][n1] = hijo1
      array_cruces[i+1][n2] = hijo2

      i += 2

    # Se vuelven a meter en la poblacion
    for hijo in array_cruces:
      progenitores.append(hijo)

    #print(len(progenitores))

    return progenitores
    

  # Cruce uniforme de una regla de ambos individuos 
  def cruce_uniforme(self, n1, n2, array_cruces, hijo1, hijo2, i):
        
    for j in range(len(array_cruces[i][n1])):
      n = random.randint(0, 1)
      # Se coge bit del primer progenitor
      if(n == 1):
        hijo1.append(array_cruces[i][n1][j])    
        hijo2.append(array_cruces[i+1][n2][j])   
      # Se coge bit del segundo progenitor      
      else:
        hijo1.append(array_cruces[i+1][n2][j])  
        hijo2.append(array_cruces[i][n1][j]) 


  # Cruce en un punto de una regla de ambos individuos 
  def cruce_un_punto(self, n1, n2, array_cruces, hijo1, hijo2, i):
    # Elegimos el punto del cruce de manera aleatoria
    n = random.randint(0, len(array_cruces[i][n1])-1)

    hijo1.extend(array_cruces[i][n1][0:n]) 
    hijo1.extend(array_cruces[i+1][n2][n:])

    hijo2.extend(array_cruces[i+1][n2][0:n])
    hijo2.extend(array_cruces[i][n1][n:])


  # Selecciona aleatoriamente, en proporcion a fitness, los progenitores de la poblacion
  def ruleta_fitness(self, fitness):
    progenitores = [] # Array progenitores
    prob_seleccion = [] # Array probabilidades seleccion individuos
    
    # Calculamos la probabilidad de seleccion para cada individuo
    sumFitness = 0.0
    for f in fitness:
      sumFitness += f
      
    for f in fitness:
      if(sumFitness == 0):
        prob_seleccion.append(0.0)
      else:
        prob_seleccion.append(f/sumFitness)

    #print(prob_seleccion)

    # Ruleta
    # Generamos un numero aleatorio entre 0.0 y 1.0
    for i in range(self.tam_poblacion):
      n = random.random()
      for j in range(self.tam_poblacion):
        n -= prob_seleccion[j]
        if(n <= 0):
          progenitores.append(self.poblacion[j].copy())
          break
    
    #print(progenitores)

    return progenitores
    

  # Mira si alguna de las reglas del individuo se activa con el dato pasado
  # Devuelve el voto por mayoria de las reglas activadas
  # Si hay empate, se devuelve 1. Si ninguna se activa, se devuelve -1 (error)
  def claseEnFuncionReglas(self, individuo, dato, diccionario):
    
    array_clases = []
    numAtributos = len(diccionario) - 1
    
    # Bucle comparando reglas con el dato
    for regla in individuo: 
      i = 0
      cumpleAtributo = 0
      # Por cada bit a 1 que cumple la regla se suma 1 a cumpleAtributo
      for bit in dato[:-1]: 
        if (regla[i] == 1 and bit == 1):
          cumpleAtributo += 1
        i += 1
      # Si cumpleAtributo coincide con el numero de atributos, se activa la regla
      if (cumpleAtributo >= numAtributos):
        array_clases.append(regla[-1])
    
    # No se activaron reglas
    # Se devuelve error
    if array_clases == []:
      return -1

    # Si se activaron reglas
    # Voto por mayoria
    else:
      # Si num:1 >= num:0 se devuelve 1
      if array_clases.count(1) >= array_clases.count(0):
        return 1
      # Si num:1 < num:0 se devuelve 0
      else:
        return 0


  # Se comparan los datos de test con el mejor individuo entrenado
  # Si cumple alguna de sus reglas, se le da esa clase 
  # Si no cumple ninguna, se da la clase de la regla mas cumplida
  def clasifica(self,datostest,atributosDiscretos,diccionario):
        
    predicciones = []

    # Convertimos los datostest a binario con el mismo formato
    datostest_sinclase = []
    for dato in datostest:
      datostest_sinclase.append(dato[0:len(dato)-1])
    enc = OneHotEncoder()
    enc.fit(datostest_sinclase)
    datostest_binarios = enc.transform(datostest_sinclase).toarray()
    
    self.datostest = []
    for dato in datostest_binarios:
      self.datostest.append(dato.tolist())

    i = 0
    for d in self.datostest:
      d.append(datostest[i][-1])
      i += 1
    
    # Bucle de clasificaciones
    for dato in self.datostest:
      # Hallamos la clase que predice el individuo sobre ese dato
      # Si no cumple ninguna regla devuelve -1 (error)
      c = self.claseEnFuncionReglas(self.mejor_individuo, dato, diccionario)
      
      # Si no hay error
      if(c != -1):
        # Comprobamos si es acierto
        predicciones.append(c)
      else:
        # Clase por defecto 1
        predicciones.append(1)
    
    return predicciones
