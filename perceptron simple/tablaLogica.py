import matplotlib.pyplot as plt
import numpy as np

class Plano:
    def __init__(self, inputs, outputs, bias, weights):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Clasificación de A ∩ (B ∪ C)")
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1, 2)
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
        
        x = np.linspace(-0.5, 1.5, 100)
        # y = (w1x/w2) - (b/w2)
        m = -(weights[0] / weights[1]) 
        b = -(bias / weights[1])
        # Calculo de la pendiente y=mx+b
        y =  m * x + b
        self.ax.plot(x, y, label="Linea de decisión", color="green")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.legend(loc = "upper right")
        plt.show()

class Perceptron:
    def __init__(self, inputs, outputs, weights = np.zeros(2), bias = 0, learning = 1):
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
        final_outputs = np.array([np.nan] * 4)
        n = len(self.inputs)
        # Repetir hasta conseguir la salida esperada
        while not np.all(final_outputs == self.outputs):
            print('Pesos acutuales y bias: ', self.weights, self.bias)
            for i in range(n):
                # Calcular potencial de activación
                # V(n) =  w1*x1 + w2*x2 + ... + wn*xn + b
                v_n = np.dot(self.weights, self.inputs[i]) + self.bias
                # Evaluar con función de activación
                y_n = self.esc(v_n)
                final_outputs[i] = y_n
                if y_n != self.outputs[i]:
                    error = self.outputs[i] - y_n
                    # Actualizar pesos y bias
                    self.adjust_weights(error, self.inputs[i])
        return final_outputs

    # Función de activación escalon
    def esc(self, x):
        if x >= 0: return 1
        else: return 0

    # Ajuste de pesos
    def adjust_weights(self, e_k, x_k):
        # w(k + 1) = w(k) + n(k)n(k)x(k)
        new_weights = self.weights + self.learning * e_k * x_k
        new_bias = self.bias + self.learning * e_k
        self.weights = new_weights
        self.bias = new_bias

if __name__ == "__main__":
    outputs = np.array([0, 1, 0, 1])
    inputs = np.array([[0, 0],
                       [1, 1],
                       [0, 1],
                       [1, 1]])
    weights = np.random.rand(2)  # Pesos aleatorios entre 0 y 1
    bias = np.random.rand() # Bias aleatorio entre 0 y 1
    perceptron = Perceptron(inputs, outputs, weights, bias) # Perceptron con valores aleatorios

    final_outputs = perceptron.calculate()
    print("Salidas finales: ", final_outputs)
    final_weights = perceptron.get_weights()
    final_bias = perceptron.get_bias()
    print("Pesos finales: ", final_weights)
    print("Bias final: ", final_bias)
    plano = Plano(inputs, final_outputs, final_bias, final_weights)