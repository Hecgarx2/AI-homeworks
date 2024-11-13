import numpy as np

# Datos de entrada y parámetros
x = np.array([0.5, 0.31, 0.4])
C = np.array([0.05, 0.7, 0.28])
d = 0.2

# Cálculo de la distancia euclidiana al cuadrado entre x y C
distance_squared = np.sum((x - C) ** 2)
print(distance_squared / ((2 * d )** 2))

# Cálculo de la activación utilizando la función gaussiana
activation = np.exp(-distance_squared / ((2 * d )** 2))
print(activation)
