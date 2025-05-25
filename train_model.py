import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from filter import (
    cargar_y_filtrar_tiendas,
    limpiar_columnas_categoricas,
    calcular_tiendas_exitosas,
    etiquetar_tiendas_exitosas
)

# Validación de archivos
for archivo in ['DIM_TIENDA.csv', 'Venta.csv', 'Meta_Venta.csv', '2024_1_19_ENT.shp', '2024_1_28_ENT.shp']:
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {archivo}")

# Cargar y procesar datos
venta = pd.read_csv('Venta.csv')
meta_venta = pd.read_csv('Meta_Venta.csv')
dim_tienda = cargar_y_filtrar_tiendas('DIM_TIENDA.csv', '2024_1_19_ENT.shp', '2024_1_28_ENT.shp')
dim_tienda = limpiar_columnas_categoricas(dim_tienda)
tiendas_exitosas = calcular_tiendas_exitosas(venta, meta_venta, dim_tienda)
df = etiquetar_tiendas_exitosas(dim_tienda, tiendas_exitosas)

# Variables para modelado
num_cols = ['MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM', 'LATITUD_NUM', 'LONGITUD_NUM']
cat_cols = ['PLAZA_CVE', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA']

# Validar columnas requeridas
for col in num_cols + cat_cols + ['vale_abrir']:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")

# Preprocesamiento
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Separar datos
X = df[num_cols + cat_cols]
y = df['vale_abrir']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenamiento y evaluación
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualización
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='LONGITUD_NUM',
    y='LATITUD_NUM',
    hue='vale_abrir',
    palette='Set1',
    alpha=0.7
)
plt.title('Ubicaciones de tiendas y su clasificación')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.grid(True)
plt.tight_layout()
plt.show()

# Guardar modelo
joblib.dump(pipeline, 'modelo_completo.pkl')