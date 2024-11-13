import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

# class Plano:
#     def __init__(self, n_inputs, n_ocultas, n_outputs):
#         # Inicializar los pesos de la capa oculta y la capa de salida
#         self.bias_oculta = np.random.uniform(0, 1, (1, n_ocultas))
#         self.pesos_salida = np.random.uniform(0, 1, (n_ocultas, n_outputs))
#         self.bias_salida = np.random.uniform(0, 1, (1, n_outputs))
#         self.error_promedio = []
    
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_title("Red de base radial")
#         self.ax.set_xlim(0, np.pi*6)
#         self.ax.set_ylim(-4, 4)
#         self.ax.grid()

#         x = np.linspace(0, np.pi*6, 100)
#         y = self.function(x)
#         self.ax.plot(x, y, label="2cos(4x) - sen(1-x)", color="green")
#         self.ax.legend(loc = "upper right")
#         plt.show()

    
#     def function(self, x):
#         return (2 * np.cos(4 * x)) - np.sin(1 - x)

# Plano()


def function(x):
        return (2 * np.cos(4 * x)) - np.sin(1 - x)

# Generar datos de ejemplo
x = np.linspace(0, np.pi*6, 100)[:, None]  # 100 puntos entre 0 y 10
y = function(x).ravel()  # Generar los valores de y = sin(x)

# Añadir algo de ruido a los datos de entrenamiento
y += 0.1 * np.random.randn(*y.shape)

# Crear un modelo Kernel Ridge con un kernel RBF (Base Radial)
model = KernelRidge(kernel='rbf', gamma=0.9)

# Entrenar el modelo con los datos de entrada y salida
model.fit(x, y)

# Generar predicciones sobre los mismos datos
y_pred = model.predict(x)

# Visualizar los datos originales y las predicciones del modelo
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Datos originales (ruidosos)')
plt.plot(x, y_pred, label='Aproximación con RBF', color='red')
plt.title("Aproximación de $2cos(4x) - sen(1-x))$ usando RBF")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

