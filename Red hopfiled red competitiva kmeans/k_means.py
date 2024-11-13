import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class K_Means:
    def __init__(self, k, data, epocas=100, tolerancia=1e-4):
        self.k = k  
        self.data = data[['Peso', 'Indice PH']].values 
        self.epocas = epocas  
        self.tolerancia = tolerancia  # Umbral de tolerancia para detener el entrenamiento
        self.centroids = np.random.rand(k, self.data.shape[1])  # Inicialización aleatoria de centroides
        self.fig, self.ax = plt.subplots()  # Crear figura y ejes una vez
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'brown']

    def entrenar(self):
        for epoch in range(self.epocas):
            centroids_previos = self.centroids.copy()  # Guardar los centroides de la iteración anterior

            # Paso 1: Asignar cada punto de datos al centroide más cercano
            clusters = [[] for _ in range(self.k)]
            for x in self.data:
                distancias = np.linalg.norm(self.centroids - x, axis=1)
                cluster_asignado = np.argmin(distancias)
                clusters[cluster_asignado].append(x)
            
            # Paso 2: Actualizar cada centroide como el promedio de los puntos asignados a él
            for i in range(self.k):
                if clusters[i]:  # Verificar que el cluster no esté vacío
                    self.centroids[i] = np.mean(clusters[i], axis=0)

            # Calcular el desplazamiento máximo de los centroides
            max_desplazamiento = np.max(np.linalg.norm(self.centroids - centroids_previos, axis=1))

            # Verificar la condición de parada
            if max_desplazamiento < self.tolerance:
                print(f"Entrenamiento detenido en la epoch {epoch+1} por convergencia.")
                break

            # Actualizar los clusters en los datos
            data['Cluster'] = self.clasificar()

            # Actualizar gráficos de manera más eficiente
            self.ax.clear()  # Limpiar el contenido de los ejes
            for cluster in range(self.k):
                cluster_data = data[data['Cluster'] == cluster]
                self.ax.scatter(cluster_data['Peso'], cluster_data['Indice PH'], 
                                marker='o', color=self.colors[cluster % len(self.colors)], label=f'Cluster {cluster+1}')
            
            # Graficar los centroides
            self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                            marker="*", s=100, color='black', label="Centroides")
            
            self.ax.set_title(f"k-means Clustering - Epoch {epoch+1}")
            self.ax.set_xlabel("Peso")
            self.ax.set_ylabel("Indice PH")
            self.ax.legend(loc="upper right")

            plt.pause(0.1)  # Reducir el tiempo de pausa para mejorar la velocidad


    def clasificar(self):
        # Asigna cada punto al centroide más cercano y devuelve los clusters
        cluster_labels = []
        for x in self.data:
            distancias = np.linalg.norm(self.centroids - x, axis=1)
            cluster_asignado = np.argmin(distancias)
            cluster_labels.append(cluster_asignado)
        return np.array(cluster_labels)

# Número de clusters
k = 6
data = pd.read_csv('datos_medicamentos.csv')

# Crear e inicializar el modelo de k-means
k_means = K_Means(k, data)

# Entrenar y visualizar
k_means.entrenar()

plt.show()
