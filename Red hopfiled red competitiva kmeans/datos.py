import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv('datos_medicamentos.csv')

# Graficar solo los puntos
plt.scatter(df['Peso'], df['Indice PH'], marker='o', color='b')

# Configurar etiquetas y título
plt.xlabel('Peso')
plt.ylabel('Indice PH')
plt.title('Gráfico de Puntos CSV')

# Mostrar la gráfica
plt.show()