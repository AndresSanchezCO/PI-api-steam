import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Parquet
data = pd.read_parquet('australian_user_reviews (1).parquet')

# EDA (Análisis Exploratorio de Datos)
# 1. Información básica sobre el conjunto de datos
print("Información básica del conjunto de datos:")
print(data.info())

# 2. Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(data.describe())

# 3. Valores faltantes
print("\nValores faltantes:")
print(data.isnull().sum())

# 4. Visualizaciones (ajustar las visualizaciones según tus necesidades)
# Por ejemplo, histograma de una columna numérica
plt.figure(figsize=(8, 4))
sns.histplot(data['tu_columna_numerica'], kde=True)
plt.title('Histograma de tu_columna_numerica')
plt.xlabel('tu_columna_numerica')
plt.ylabel('Frecuencia')
plt.show()

# ETL (Extracción, Transformación y Carga)
# 1. Eliminar filas con valores nulos (ajustar según necesidades)
data = data.dropna()

# 2. Eliminar outliers (ajustar según necesidades)
def remove_outliers_zscore(df, columns, z_threshold=3):
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[(z_scores < z_threshold)]
    return df

# Especifica las columnas numéricas en las que deseas eliminar outliers
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = remove_outliers_zscore(data, numeric_columns)

# 3. Guardar el resultado en un nuevo archivo Parquet
data.to_parquet('cleaned_australian_user_reviews.parquet', index=False)

print("Proceso de EDA y ETL completado. Los datos limpios se han guardado en 'cleaned_australian_user_reviews.parquet'.")
