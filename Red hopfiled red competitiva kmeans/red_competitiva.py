import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
inicio = time.time()


class RedCompetitiva:
    def __init__(self, k, data, learning_rate=0.5, epochs=100):
        self.k = k  # Número de neuronas en la capa competitiva
        self.data = data[['Peso', 'Indice PH']].values  # Datos de entrada
        self.learning_rate = learning_rate  # Tasa de aprendizaje
        self.epochs = epochs  # Número de iteraciones
        self.weights = np.random.rand(k, self.data.shape[1])  # Pesos iniciales aleatorios

    def entrenar(self):
        for epoch in range(self.epochs):
            for x in self.data:
                # Paso 1: Encuentra la neurona ganadora
                distancias = np.linalg.norm(self.weights - x, axis=1)
                ganadora = np.argmin(distancias)
                
                # Paso 2: Actualiza los pesos de la neurona ganadora
                self.weights[ganadora] += self.learning_rate * (x - self.weights[ganadora])

    def clasificar(self):
        clusters = []
        for x in self.data:
            distancias = np.linalg.norm(self.weights - x, axis=1)
            ganadora = np.argmin(distancias)
            clusters.append(ganadora)
        return np.array(clusters)

class Plano:
    def __init__(self, data, model):
        self.fig, self.ax = plt.subplots()
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'brown']
        
        for cluster in range(model.k):
            cluster_data = data[data['Cluster'] == cluster]
            self.ax.scatter(cluster_data['Peso'], cluster_data['Indice PH'], 
                    marker='o', color=self.colors[cluster % len(self.colors)], label=f'Cluster {cluster+1}')
        
        self.ax.scatter(model.weights[:,0], model.weights[:,1], 
                        marker="*", s=100, color='black', label="Neurona")
        
        self.ax.set_title("Red Neuronal Competitiva")
        self.ax.set_xlabel("Peso")
        self.ax.set_ylabel("Indice PH")
        self.ax.legend(loc = "upper right")
        plt.show()

# Número de neuronas en la capa competitiva
k = 4
data = pd.read_csv('datos_medicamentos.csv')

# Crear e inicializar la red neuronal competitiva
red_competitiva = RedCompetitiva(k, data)
red_competitiva.entrenar()
data['Cluster'] = red_competitiva.clasificar()

fin = time.time()
print(fin-inicio) 
# Mostrar el gráfico
Plano(data, red_competitiva)
