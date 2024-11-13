import matplotlib.pyplot as plt
import numpy as np

patrones = np.array([
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


fig, axes = plt.subplots(2, 5, figsize=(15, 5))
fig.patch.set_facecolor('gray')  # Fondo de toda la figura

matriz0 = np.reshape(patrones[0], (7, 5))
matriz1 = np.reshape(patrones[1], (7, 5))
matriz2 = np.reshape(patrones[2], (7, 5))
matriz3 = np.reshape(patrones[3], (7, 5))
matriz4 = np.reshape(patrones[4], (7, 5))
matriz5 = np.reshape(patrones[5], (7, 5))
matriz6 = np.reshape(patrones[6], (7, 5))
matriz7 = np.reshape(patrones[7], (7, 5))
matriz8 = np.reshape(patrones[8], (7, 5))
matriz9 = np.reshape(patrones[9], (7, 5))

axes[0, 0].imshow(matriz0, cmap='binary')

axes[0, 1].imshow(matriz1, cmap='binary')

axes[0, 2].imshow(matriz2, cmap='binary')

axes[0, 3].imshow(matriz3, cmap='binary')

axes[0, 4].imshow(matriz4, cmap='binary')

axes[1, 0].imshow(matriz5, cmap='binary')

axes[1, 1].imshow(matriz6, cmap='binary')

axes[1, 2].imshow(matriz7, cmap='binary')

axes[1, 3].imshow(matriz8, cmap='binary')

axes[1, 4].imshow(matriz9, cmap='binary')

plt.show()

