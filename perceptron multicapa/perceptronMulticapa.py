import matplotlib.pyplot as plt
import numpy as np

class Plano:
    def __init__(self, inputs, outputs, bias, weights, bias_oculta, weights_oculta):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Perceptron multicapa")
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1, 2)
        self.ax.grid()
        for i in range(len(inputs)):
            if round(outputs[i][0]) == 1:
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
        y =  -(weights[0] * x + bias) / weights[1]
        ocutla1 =  -(weights_oculta[0][0] * x + bias_oculta[0][0]) / weights_oculta[0][1]
        ocutla2 =  -(weights_oculta[1][0] * x + bias_oculta[0][1]) / weights_oculta[1][1]

        # Calculo de la pendiente y=mx+b
        # self.ax.plot(x, y[0], label="Linea de decisión", color="red")
        self.ax.plot(x, ocutla1, label="Linea de decisión", color="green")
        self.ax.plot(x, ocutla2, color="green")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.legend(loc = "upper right")
        plt.show()

def plot_Errores(errores):
    plt.figure(figsize=(8, 6))
    plt.plot(errores, label='MSE', color='b')
    plt.title('Errores Cuadráticos Medios')
    plt.xlabel('Iteraciones (Épocas)')
    plt.ylabel('MSE (Error Cuadrático Medio)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Definir la función sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Definir la red neuronal con una capa oculta
class PerceptronMulticapa:
    def __init__(self, n_inputs, n_ocultas, n_outputs):
        # Inicializar los pesos de la capa oculta y la capa de salida
        self.pesos_oculta = np.random.uniform(0, 1, (n_inputs, n_ocultas))
        self.bias_oculta = np.random.uniform(0, 1, (1, n_ocultas))
        self.pesos_salida = np.random.uniform(0, 1, (n_ocultas, n_outputs))
        self.bias_salida = np.random.uniform(0, 1, (1, n_outputs))
        self.error_promedio = []
    
    def propagacion(self, x):
        # Cálculo de la capa oculta
        v_n = np.dot(x, self.pesos_oculta) + self.bias_oculta
        self.y_oculta = sigmoid(v_n)
        
        # Cálculo de la capa de salida
        salida = sigmoid(np.dot(self.y_oculta, self.pesos_salida) + self.bias_salida)
        return salida

    def retropropagacion(self, x, y, salida, tasa_aprendizaje=0.75):
        # Error de la capa de salida
        # e(k) = d(k) - y(k) Salida esperada menos salida obtenida
        error_salida = y - salida
        self.error_promedio.append(np.mean(np.square(error_salida)))
        # δ_n(k) = e(k)p'(v(k)) Delta igual al error por la derivada de la funcion de activacion
        delta_salida = error_salida * sigmoid_derivative(salida)

        # Error de la capa oculta
        # e_j(k) = Σ W_nj(k)δ_n(k) Error de la capa ocutla igual a la sumatoria de los pesos por el delta de salida
        error_oculta = delta_salida.dot(self.pesos_salida.T)
        # δ_j(k) = p'(v(k))e_j(k) Delta de capa oculta igual a derivada de la funcion de activacion por el 
        # error de la capa oculta
        delta_oculta = error_oculta * sigmoid_derivative(self.y_oculta)

        # W = η * δ(k) * y(k)
        # Ajustar los pesos de la capa de salida
        self.pesos_salida += tasa_aprendizaje * self.y_oculta.T.dot(delta_salida) 
        self.bias_salida += tasa_aprendizaje * np.sum(delta_salida, axis=0, keepdims=True)

        # Ajustar los pesos de la capa oculta
        self.pesos_oculta += tasa_aprendizaje * x.T.dot(delta_oculta)
        self.bias_oculta += tasa_aprendizaje * np.sum(delta_oculta, axis=0, keepdims=True)

# Datos de ejemplo
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])  # XOR

# Crear y entrenar la red
red = PerceptronMulticapa(n_inputs=2, n_ocultas=2, n_outputs=1)
epochs = 2000
for _ in range(epochs):
    salida = red.propagacion(inputs)
    red.retropropagacion(inputs, outputs, salida)
# Prueba la red después de entrenar
print("Salida después de entrenar:")
print(red.propagacion(inputs))
plot_Errores(red.error_promedio)
plano = Plano(inputs, red.propagacion(inputs), red.bias_salida, red.pesos_salida, 
              red.bias_oculta, red.pesos_oculta)

