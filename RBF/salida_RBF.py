# Pesos y bias
w1 = 0.7
w2 = 0.8
w3 = 0.9
bias = 1

# Activaciones
phi1 = 0.57335  # Activación de la primera neurona
phi2 = 0.06465  # Activación de la segunda neurona
phi3 = 0.09963 # Activación de la tercera neurona

# Cálculo de la salida
output = (w1 * phi1 + w2 * phi2 + w3 * phi3) + bias
print(output)
