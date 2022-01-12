# -*- coding: ISO-8859-1 -*-
# =============================================================================
# Ruta del archivo. Se debe llamar "data" y debe estar en la misma carpeta
# =============================================================================
file = 'data.csv'

# =============================================================================
# Leer archivos CSV (comma separated values) con pandas
# =============================================================================
import pandas as pd

df = pd.read_csv(file)
print("\n{0:=^75}".format(" Lectura de csv completa "))

print("Dimensiones:"
     +f"\n\tNumero de registros: {df.shape[0]}"
     +f"\n\tNumero de variables: {df.shape[1]}\n")

print("Nombres y tipos de datos:")
print(df.dtypes)

# Quita los renglones (axis=0) que contienen cualquier columna vacía
df.dropna(axis = 0, how = 'any', inplace = True)

# =============================================================================
# Analisis de datos por columna. 
# Las columnas seleccionadas son energy y danceability
# =============================================================================
def mostrar_datos(nombre):
    if(nombre in list(df.columns)):
        columna = df[nombre.lower()]
        print(f"\n{columna.head().name:-^75}")
        print(f"Valores unicos:\n{columna.unique()}")
        print(f"\nValor maximo:\t\t{columna.max()}")
        print(f"Valor minimo:\t\t{columna.min()}")
        print(f"Media:\t\t\t{columna.mean()}")
        print(f"Mediana:\t\t{columna.median()}")
        print(f"Desviacion estandar:\t{columna.std()}")
    else:
        print(f"\nColumna '{nombre}' no encontrada")

# =============================================================================
# Histograma
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def reporte(variable1, variable2):
    print("\n{0:=^75}".format(" Comparativa entre " + variable1 + " y " + variable2 + " "))
    mostrar_datos(variable1)
    mostrar_datos(variable2)

    colors = ['coral', 'lightseagreen']
    data = list(df[variable1]) + list(df[variable2])
    bins = np.arange(0, max(data) + 2.5) - min(data) - 0.5
    plt.style.use('seaborn-deep')
    
    fig, ax1 = plt.subplots(1, 1)
    ax1.hist([df[variable1], df[variable2]], bins, edgecolor = 'black', 
            color = colors, label = [variable1, variable2])
    
    # Set title
    ax1.set_title("Histograma")
     
    # adding labels
    ax1.set_xlabel('valor de registro')
    ax1.set_ylabel('cantidad de registros')
    
    plt.legend(loc = "upper right")
    plt.show()
    
    # =========================================================================
    # Cajas y bigotes
    # =========================================================================
    fig, ax2 = plt.subplots(1, 1)
    bp = ax2.boxplot([df[variable1], df[variable2]], notch=True, vert=False,
             labels=[variable1, variable2], showfliers=False, showmeans=True, 
             patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_title("Diagrama de cajas y bigotes")
    plt.show()

# =============================================================================
# Mapa de calor
# =============================================================================
plt.figure(figsize=(15, 5))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, vmin=-0.9, vmax=0.9, 
            annot=True, cmap='summer')
plt.title("Mapa de calor de correlacion")
plt.show()

# =============================================================================
# Pruebas
# =============================================================================
reporte("acousticness", "energy")
#reporte("danceability", "energy")
#reporte("loudness", "energy")
#reporte("danceability", "instrumentalness")
