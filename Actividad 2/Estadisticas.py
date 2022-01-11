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
print("\n{0:=^60}".format(" Lectura de csv completa "))

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
        print(f"\n{columna.head().name:-^60}")
        print(f"Valores unicos:\n{columna.unique()}")
        print(f"\nValor maximo:\t\t{columna.max()}")
        print(f"Valor minimo:\t\t{columna.min()}")
        print(f"Media:\t\t\t{columna.mean()}")
        print(f"Mediana:\t\t{columna.median()}")
        print(f"Desviacion estandar:\t{columna.std()}")
    else:
        print(f"\nColumna '{nombre}' no encontrada")

mostrar_datos("energy")
mostrar_datos("danceability")