# -*- coding: ISO-8859-1 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

# =============================================================================
# Ruta del archivo. Se debe llamar "data" y debe estar en la misma carpeta
# =============================================================================
file = 'data.csv'

# Constante para marcadores en matplotlib
markers = "o v ^ < > 1 2 3 4 8 s p P * h H + x X D d | _".split()

# =============================================================================
# Leer archivos CSV (comma separated values) con pandas
# =============================================================================
df = pd.read_csv(file)
print("\n{0:=^75}".format(" Lectura de csv completa "))

print("Dimensiones:"
     +f"\n\tNumero de registros: {df.shape[0]}"
     +f"\n\tNumero de variables: {df.shape[1]}\n")

print("Nombres y tipos de datos:")
print(df.dtypes)

# Quita los renglones (axis=0) que contienen cualquier columna vacía
df.dropna(axis = 0, how = 'any', inplace = True)

def mostrar_datos(nombre):
    if(nombre in list(df.columns)):
        columna = df[nombre.lower()]
        print("\n{0:-^75}".format(f" Datos de {columna.head().name} "))
        print(df[nombre].describe(), "\n")
        print(f"Valores unicos:\n{columna.unique()}")
        print(f"\nValor maximo:\t\t{columna.max()}")
        print(f"Valor minimo:\t\t{columna.min()}")
        print(f"Media:\t\t\t{columna.mean()}")
        print(f"Mediana:\t\t{columna.median()}")
        print(f"Desviacion estandar:\t{columna.std()}")
    else:
        print(f"\nColumna '{nombre}' no encontrada")

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
# Analisis de datos por columna
# =============================================================================
class Modelo:
    
    def __init__(self, variable1, variable2):
        self.variable1 = variable1
        self.variable2 = variable2
        self.comparativa()
    
    def comparativa(self):
        print("\n{0:=^75}".format(" Comparativa entre " + self.variable1 
                                  + " y " + self.variable2 + " "))
        mostrar_datos(self.variable1)
        mostrar_datos(self.variable2)
        print("\n{0:-^75}".format(f" Diagramas y graficas comparativas "))
        # =====================================================================
        # Histograma
        # =====================================================================
        colors = ['coral', 'lightseagreen']
        data = list(df[self.variable1]) + list(df[self.variable2])
        bins = np.arange(0, max(data) + 2.5) - min(data) - 0.5
        plt.style.use('seaborn-deep')
        
        fig, ax1 = plt.subplots(1, 1)
        ax1.hist([df[self.variable1], df[self.variable2]], bins, 
                 edgecolor = 'black', color = colors, 
                 label = [self.variable1, self.variable2])
        
        # Set title
        ax1.set_title("Histograma")
         
        # adding labels
        ax1.set_xlabel('valor de registro')
        ax1.set_ylabel('cantidad de registros')
        
        plt.legend(loc = "upper right")
        plt.show()
        
        # =====================================================================
        # Cajas y bigotes
        # =====================================================================
        fig, ax2 = plt.subplots(1, 1)
        bp = ax2.boxplot([df[self.variable1], df[self.variable2]], 
                         notch=True, vert=False, showfliers=False, 
                         labels=[self.variable1, self.variable2], 
                         showmeans=True, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title("Diagrama de cajas y bigotes")
        ax2.set_xlabel('valor de registro')
        plt.show()
        
        self.k_means(3)
    
    def k_means(self, k, valor1 = None, valor2 = None):
        # =====================================================================
        # K means
        # =====================================================================
        test = df[[self.variable1, self.variable2]]
        test = test.dropna(axis = 0, how = 'any')
        kmeans = KMeans(n_clusters = k).fit(test)
        centroids = kmeans.cluster_centers_
        
        model_colours = kmeans.predict(test)
        plt.scatter(df[self.variable1], df[self.variable2], alpha = 0.5, 
                    c = model_colours)
        for i in range(len(centroids)):
            plt.scatter(centroids[i][0], centroids[i][1], marker = markers[i], 
                        c = "red", label = f"Grupo {i + 1}", s = 80)
        
        if(valor1 != None):
            print("\n{0:-^75}".format(" Prediccion para el "
                                     + f"valor {(valor1, valor2)} "))
            data = {self.variable1: [valor1], self.variable2: [valor2]}
            newdf = pd.DataFrame(data)
            print(f"Grupo al que pertenece: {kmeans.predict(newdf)[0] + 1}")
            plt.scatter(valor1, valor2, marker = "*", 
                        c = "midnightblue", s = 100)
        
        
        plt.xlabel(self.variable1)
        plt.ylabel(self.variable2)
        plt.legend(loc = "upper right")
        plt.title("Mapa de K-means del modelo")

        self.plot_regression_line(self.estimate_coef())


    def estimate_coef(self):
        # Codigo proporcionado por Geeksforgeeks: 
        # https://www.geeksforgeeks.org/linear-regression-python-implementation/
        
        x = df[self.variable1]
        y = df[self.variable2]
        
        # number of observations/points
        n = np.size(x)
     
        # mean of x and y vector
        m_x = np.mean(x)
        m_y = np.mean(y)
     
        # calculating cross-deviation and deviation about x
        SS_xy = np.sum(y * x) - n * m_y * m_x
        SS_xx = np.sum(x * x) - n * m_x * m_x
     
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
        
        return (b_0, b_1)
     
    def plot_regression_line(self, b):
        # =====================================================================
        # Regresion linear
        # =====================================================================
        # Codigo proporcionado por Geeksforgeeks: 
        # https://www.geeksforgeeks.org/linear-regression-python-implementation/

        # predicted response vector
        y_pred = b[0] + b[1] * df[self.variable1]

        # plotting the regression line
        plt.plot(df[self.variable1], y_pred, color = "g")

        # function to show plot
        plt.show()
        
        print(f"\nCoeficientes de la recta:\n\tInterseccion del eje y:\t {b[0]}"
             +f"\n\tPendiente: \t\t{b[1]}")

# =============================================================================
# Ejecucion y pruebas
# =============================================================================
modelo = Modelo("acousticness", "energy")

modelo.k_means(3, 0.5, 0.5)