# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga el conjunto de datos
data = pd.read_csv('output_steam_games.csv')

# Información básica sobre el conjunto de datos
print("Información básica del conjunto de datos:")
print(data.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(data.describe())

# Valores faltantes
print("\nValores faltantes:")
print(data.isnull().sum())

# Histogramas de variables numéricas
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

# Box plots de variables numéricas
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col])
    plt.title(f'Box Plot de {col}')
    plt.xlabel(col)
    plt.show()

# Gráficos de barras de variables categóricas
categorical_cols = data.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data[col], order=data[col].value_counts().index)
    plt.title(f'Gráfico de Barras de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

# Matriz de correlación
correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()
