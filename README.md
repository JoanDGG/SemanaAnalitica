# SemanaAnalitica
Repositorio para actividades de la Semana Tec de Herramientas computacionales: el arte de la analítica.

## Como correr el programa
Para ejecutar ```Estadisticas.py``` debe tener el archivo ```data.csv``` dentro de la misma carpeta y se puede correr utilizando el siguiente comando

```
python Estadisticas.py
```
El programa ya lee el archivo csv al inicializar el programa, para realizar un análisis comparativo entre dos variables, es necesario crear un objeto de tipo Modelo, el cual recibe dos cadenas de texto con las variables a analizar, esto entrega de manera automática un reporte con ambas variables.

Adicionalmente, puede utilizar las funciones que requiera, ya sea para un análisis de datos individual o predictivo:

- Para revisar la información de una columna, se puede llamar a la funcion mostrar_datos() con el nombre de la columna deseada
- Para hacer una predicción de un dato con el modelo generado, es necesario llamar a la función k_means() del modelo que haya creado, este requiere de el valor de k, y las coordenadas x y y del valor a analizar.

## Consideraciones
Dada la naturaleza del archivo ```data.csv```, todos los elementos dentro de este deben ser valores numéricos.

# Autor
Joan Daniel Guerrero García A01378052
