import pandas as pd
import numpy as np
from scipy import stats

# Cargar el conjunto de datos
data = pd.read_csv('output_steam_games.csv')

# Eliminar filas con valores nulos
data = data.dropna()

# Definir una función para detectar y eliminar outliers utilizando el método Z-score
def remove_outliers_zscore(df, columns, z_threshold=3):
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[(z_scores < z_threshold)]
    return df

# Especificar las columnas numéricas en las que deseas eliminar outliers
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = remove_outliers_zscore(data, numeric_columns)

# Guardar el resultado en un nuevo archivo CSV
data.to_csv('cleaned_steam_games.csv', index=False)

print("Proceso de ETL completado. Los datos limpios se han guardado en 'cleaned_steam_games.csv'.")
