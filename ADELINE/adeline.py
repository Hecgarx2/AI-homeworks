import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class Plano:
    def __init__(self, inputs, outputs, bias, weights):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Adeline")
        self.ax.set_xlim(-.5, 1.5)
        self.ax.set_ylim(-.5, 1.5)
        self.ax.grid()

        for i in range(len(inputs)):
            if outputs[i] == 1:
                self.ax.scatter(inputs[i][0], inputs[i][1], color='blue', marker='s',  label='Clase 1' 
                                if 'Clase 1' not in self.ax.get_legend_handles_labels()[1] else "")
                self.ax.annotate('({},{})'.format(inputs[i][0], inputs[i][1]),
                                 xy=(inputs[i][0], inputs[i][1]))
            else:
                self.ax.scatter(inputs[i][0], inputs[i][1], color='red', label='Clase 0'
                                if 'Clase 0' not in self.ax.get_legend_handles_labels()[1] else "")
                self.ax.annotate('({},{})'.format(inputs[i][0], inputs[i][1]),
                                 xy=(inputs[i][0], inputs[i][1]))
        
        x = np.linspace(-.5, 1.5)
        # y = (w1x/w2) - (b/w2)
        m = -(weights[0] / weights[1]) 
        b = -(bias / weights[1])
        # Calculo de la pendiente y=mx+b
        y =  (m * x + b)
        self.ax.plot(x, y, label="Linea de descion", color="green")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.legend(loc = "upper right")
        plt.show()

class Adeline:
    def __init__(self, inputs, outputs, weights = np.array([0.1, 0.1]), bias = 0.1, learning = 0.1):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs
        self.outputs = outputs
        # Rango de aprendizaje
        self.learning = learning
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias

    def calculate(self):
        bin_outputs = np.array([np.nan] * 4)
        real_outputs = np.array([np.nan] * 4)
        n = len(self.inputs)
        iter = 0
        # Repetir hasta conseguir la salida esperada
        while not np.all(bin_outputs == self.outputs) and iter != 30:
            for i in range(n):
                # Calcular potencial de activación
                # V(n) =  w1*x1 + w2*x2 + ... + wn*xn + b
                v_n = np.dot(self.weights, self.inputs[i]) + self.bias

                real_outputs[i] = v_n
                if v_n != self.outputs[i]:
                    error = self.outputs[i] - v_n
                    # Actualizar pesos y bias
                    delta = self.delta(v_n, self.outputs[i])
                    self.adjust_weights(error, self.inputs[i], delta)
                    print('Pesos acutuales y bias: ', self.weights, self.bias)
                # Una vez ajustado el peso la salida v_n se pasa por un clasificador
                bin_outputs[i] = self.sign(v_n)
            iter += 1
        return real_outputs, bin_outputs
       
    # Función de activación signo
    def sign(self, x):
        if x > 0: return 1
        elif x < 0: return -1
        else: return 0

    # Función delta ajuste de pesos
    def delta(self, y_k, d_k):
        # d(k) = salida esperada
        # y(k) = salida obtenida
        # Si y(k) es menor que d(k) entonces el peso se incrementa
        # Si y(k) es mayor que d(k) entonces el peso se decrementa
        # Si y(k) es igual que d(k) entonces el peso no se modifica
        if d_k > y_k: return 1
        elif d_k < y_k: return -1
        else: return 0

    # Ajuste de pesos
    def adjust_weights(self, e_k, x_k, delta):
        # w(k + 1) = w(k) + n(k)n(k)x(k)
        new_weights = self.weights + delta * self.learning * x_k * e_k
        new_bias = self.bias + self.learning * e_k
        self.weights = new_weights
        self.bias = new_bias

if __name__ == "__main__":
    outputs = np.array([0, 1, 1, 1])
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    # weights = np.random.rand(2)  # Pesos aleatorios entre 0 y 1
    # bias = np.random.rand() # Bias aleatorio entre 0 y 1
    # adeline = Adeline(inputs, outputs, weights, bias) # Perceptron con valores aleatorios

    adeline = Adeline(inputs, outputs)
    real_outputs, bin_outputs = adeline.calculate()
    print("Salidas reales: ", real_outputs)
    print("Salidas binarias: ", bin_outputs)
    final_weights = adeline.get_weights()
    final_bias = adeline.get_bias()
    print("Pesos finales: ", final_weights)
    print("Bias final: ", final_bias)
    plano = Plano(inputs, bin_outputs, final_bias, final_weights)