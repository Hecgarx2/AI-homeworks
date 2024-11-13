import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import time
inicio = time.time()

class Plano:
    def __init__(self, data, kmeans):
        self.fig, self.ax = plt.subplots()
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'brown']

        for cluster in range(kmeans.n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            self.ax.scatter(cluster_data['Peso'], cluster_data['Indice PH'], 
                            marker='o', color=self.colors[cluster], label=f'Cluster {cluster+1}')

        self.ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                        marker="*", s=100, color='black', label="Centroides")

        self.ax.set_title("k-means")
        self.ax.set_xlabel("Peso")
        self.ax.set_ylabel("Indice PH")
        self.ax.legend(loc = "upper right")
        plt.show()

class K_Means:
    def __init__(self, k, data):
        self.k = k
        self.data = data[['Peso', 'Indice PH']]
        self.kmeans = KMeans(n_clusters=k, n_init=10).fit(self.data.values)

    def get_labels(self):
        return self.kmeans.labels_
    
# Clusters
k = 6
data = pd.read_csv('datos_medicamentos.csv')
print(data)
kmeans_instance = K_Means(k, data)
data['Cluster'] = kmeans_instance.get_labels()

fin = time.time()
print(fin-inicio) 
# Mostrar el gráfico
Plano(data, kmeans_instance.kmeans)
