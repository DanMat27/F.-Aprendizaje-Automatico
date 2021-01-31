# -*- coding: utf-8 -*-
'''
Práctica 1
Autores: Laura Sanchez Herrera
         Daniel Mateo Moreno
Pareja: 02
Grupo: 1462
'''
import pandas as pd
import numpy as np


class Datos:

  # Inicializacion de la clase Datos
  def __init__(self, nombreFichero):
    # Atributos de la clase Datos
    self.nominalAtributos = []
    self.datos = []
    self.diccionario = {}

    # Leemos el fichero y creamos el DataFrame
    data = pd.read_csv(nombreFichero, dtype={'Pclass':'object', 'Age':'object'})

    # Obtenemos el tipo de cada columna
    tipos = data.dtypes

    # Indicamos si hay que realizar conversion segun el tipo de los datos leidos
    for x in tipos:
      if x == 'object':
        self.nominalAtributos.append(True)
      elif x == 'float64' or x == 'float32':
        self.nominalAtributos.append(False)
      elif x == 'int64' or x == 'int32':
        self.nominalAtributos.append(False)
      else:
        raise ValueError("Dato no contemplado!")

    #print(nominalAtributos)

    nombre_atributos = list(data.columns)

    # Creamos el diccionario que guarda las conversiones de datos
    i = 0
    for x in nombre_atributos:
      aux = []
      self.diccionario[x] = {}

      # Si hay conversion (Nominal = True)
      if self.nominalAtributos[i] is True:
        # Obtenemos los distintos valores que toma el atributo
        for fila in data.values:
          if fila[i] not in aux:
            aux.append(fila[i])

        # Ordenamos alfabeticamente los valores
        aux.sort()

        # Metemos en el diccionario las conversiones (valorAntiguo : nuevoValor)
        cont = 0
        for val in aux:
          self.diccionario[x][val] = cont
          cont += 1

      i += 1

    #print(self.diccionario)

    # Realizamos la conversion de la tabla de datos y se almacena
    array_tabla = []
    for fila in data.values:
      aux_list = []
      i = -1
      for val in fila:
        i += 1
        if self.diccionario[nombre_atributos[i]] != {}:
          convers = self.diccionario[nombre_atributos[i]].items()
          for t in convers:
            if t[0] == val:
              aux_list.append(t[1])
        else:
          aux_list.append(val)

      array_tabla.append(aux_list)

    self.datos = np.array(array_tabla)

    #Consideramos que Class es siempre Nominal para facilitar más adelante
    if self.nominalAtributos[len(self.nominalAtributos)-1] == False:
        aux = []
        self.diccionario['Class'] = {}
        self.nominalAtributos[len(self.nominalAtributos)-1] = True
        # Obtenemos los distintos valores que toma el atributo
        for fila in data.values:
          if fila[len(self.nominalAtributos)-1] not in aux:
            aux.append(fila[len(self.nominalAtributos)-1])

        # Ordenamos alfabeticamente los valores
        aux.sort()

        # Metemos en el diccionario sus valores (valor : valor)
        for val in aux:
          self.diccionario['Class'][val] = val

    #print(self.datos)

  # Genera submatriz de datos segun la lista de indices pedidos
  def extraeDatos(self, idx):

      array_submatriz = []

      # Extraemos las filas de los indices en idx y las metemos en la submatriz
      for i in idx:
          array_submatriz.append(self.datos[i])

      return np.array(array_submatriz)
