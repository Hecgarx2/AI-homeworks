import matplotlib.pyplot as plt
import numpy as np

class Plano:
    def __init__(self, n_inputs, n_ocultas, n_outputs):
        # Inicializar los pesos de la capa oculta y la capa de salida
        self.bias_oculta = np.random.uniform(0, 1, (1, n_ocultas))
        self.pesos_salida = np.random.uniform(0, 1, (n_ocultas, n_outputs))
        self.bias_salida = np.random.uniform(0, 1, (1, n_outputs))
        self.error_promedio = []
    
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Red de base radial")
        self.ax.set_xlim(0, np.pi*6)
        self.ax.set_ylim(-4, 4)
        self.ax.grid()

        x = np.linspace(0, np.pi*6, 100)
        y = self.function(x)
        self.ax.plot(x, y, label="2cos(4x) - sen(1-x)", color="green")
        self.ax.legend(loc = "upper right")
        plt.show()

    
    def function(self, x):
        return (2 * np.cos(4 * x)) - np.sin(1 - x)

Plano()