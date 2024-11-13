import matplotlib.pyplot as plt
import numpy as np

class RedHopfield:
    def __init__(self):
        self.patrones = np.array([
           [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 
                    1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
           [-1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1,
                                -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
           [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1,  
                                -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1],
           [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 
                                -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
           [-1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 
                                1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
           [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 
                                -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
           [-1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 
                                1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
           [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 
                                -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1],
           [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 
                                1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
           [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 
                                -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1]
        ])

        self.W = np.zeros((len(self.patrones[0]),len(self.patrones[0])))
        for p in self.patrones:
            self.W += np.outer(p, p)

        np.fill_diagonal(self.W, 0)
        self.error = np.inf
        self.mejor = np.zeros(len(self.patrones[0]))

        # self.W /= len(self.patrones)

    def sign(self, x):
        return np.where(x >= 0, 1, -1)

    def ruido(self, numero):
        porcentaje_ruido = 0.2  # 20% de ruido

        # Calcular el número de elementos a cambiar
        num_cambios = int(len(self.patrones[numero]) * porcentaje_ruido)

        # Seleccionar índices aleatorios para cambiar
        indices_a_cambiar = np.random.choice(len(self.patrones[numero]), num_cambios, replace=False)

        # Agregar ruido invirtiendo el signo en los índices seleccionados
        vector_con_ruido = self.patrones[numero].copy()
        vector_con_ruido[indices_a_cambiar] *= -1

        return vector_con_ruido
    
    def distancia_hamming(self, original, resultado):
        # Calcula la distancia de Hamming entre dos vectores
        return np.sum(original != resultado)
    
    def agregar_ruido(self, vector, nivel_ruido):
        # Modifica el vector con un porcentaje de elementos cambiados aleatoriamente
        vector_ruido = np.copy(vector)
        for i in range(len(vector)):
            if np.random.rand() < nivel_ruido:  # Nivel de ruido como probabilidad de cambio
                vector_ruido[i] *= -1  # Invierte el valor (de -1 a 1 o de 1 a -1)
        return vector_ruido

    def calculate(self, numero):
        max_iterations = 30000
        entrada = self.ruido(numero)
        matriz_original = np.reshape(self.patrones[numero], (7, 5))
        matriz_ruido = np.reshape(entrada, (7, 5))

        for iteration in range(max_iterations):
            # Calcular la salida de la red
            salida = self.sign(np.dot(self.W, entrada))
            
            # Verificar convergencia
            if np.array_equal(self.patrones[numero], salida):
                print("Convergió después de", iteration + 1, "iteraciones.")
                break

            error = self.distancia_hamming(self.patrones[numero], salida)
            if error < self.error:
                self.error = error
                self.mejor = salida
            
            # Determina el nivel de ruido (por ejemplo, una fracción del error actual)
            nivel_ruido = min(1.0, error / len(entrada))      
            # Actualizar la entrada
            entrada = self.agregar_ruido(salida, nivel_ruido)

        resultado = salida
        error_salida = self.distancia_hamming(self.patrones[numero], resultado)
        error_mejor = self.distancia_hamming(self.patrones[numero], self.mejor)
        if error_salida < error_mejor:
            matriz_final = np.reshape(resultado, (7, 5))
        else:
            matriz_final = np.reshape(self.mejor, (7, 5))

        # print(resultado)
        # print("Convergió en el máximo de operaciones.")
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        fig.patch.set_facecolor('gray')  # Fondo de toda la figura

        # Mostrar la primera matriz con color de fondo en los ejes
        axes[0].imshow(matriz_original, cmap='binary')
        axes[0].set_title('Numero original')
        axes[0].axis('off')

        # Mostrar la segunda matriz con color de fondo en los ejes
        axes[1].imshow(matriz_ruido, cmap='binary')
        axes[1].set_title('Numero con ruido')
        axes[1].axis('off')

        # Mostrar la segunda matriz con color de fondo en los ejes
        axes[2].imshow(matriz_final, cmap='binary')
        axes[2].set_title('Resultado')
        axes[2].axis('off')

        print(matriz_original)
        print(matriz_final)

        plt.tight_layout()
        plt.show()


red = RedHopfield()
red.calculate(9)