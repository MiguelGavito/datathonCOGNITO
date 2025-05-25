# file: test.py
import pandas as pd
import joblib
import os

# Columnas esperadas en el archivo CSV
EXPECTED_COLUMNS = [
    'TIENDA_ID', 'PLAZA_CVE', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES',
    'MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM',
    'LATITUD_NUM', 'LONGITUD_NUM', 'SEGMENTO_MAESTRO_DESC',
    'LID_UBICACION_TIENDA', 'DATASET'
]

# Columnas requeridas por el modelo
FEATURE_COLUMNS = [
    'PLAZA_CVE', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES',
    'MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM',
    'LATITUD_NUM', 'LONGITUD_NUM', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA'
]

# Tipos esperados por columna
EXPECTED_TYPES = {
    'TIENDA_ID': int,
    'PLAZA_CVE': str,
    'NIVELSOCIOECONOMICO_DES': str,
    'ENTORNO_DES': str,
    'MTS2VENTAS_NUM': float,
    'PUERTASREFRIG_NUM': float,
    'CAJONESESTACIONAMIENTO_NUM': float,
    'LATITUD_NUM': float,
    'LONGITUD_NUM': float,
    'SEGMENTO_MAESTRO_DESC': str,
    'LID_UBICACION_TIENDA': str,
    'DATASET': str
}


def validar_csv(df):
    errores = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            errores.append(f"Columna faltante: {col}")

    if not errores:
        for index, row in df.iterrows():
            for col, expected_type in EXPECTED_TYPES.items():
                val = row[col]
                if pd.isnull(val):
                    errores.append(f"Valor nulo en fila {index}, columna {col}")
                    continue
                try:
                    expected_type(val)
                except Exception:
                    errores.append(f"Tipo inválido en fila {index}, columna {col}: {val}")

            try:
                lat = float(row['LATITUD_NUM'])
                lon = float(row['LONGITUD_NUM'])
                if not (14.5 <= lat <= 33.0):
                    errores.append(f"Latitud fuera de rango en fila {index}: {lat}")
                if not (-118.0 <= lon <= -86.0):
                    errores.append(f"Longitud fuera de rango en fila {index}: {lon}")
            except Exception:
                errores.append(f"Latitud o longitud inválida en fila {index}")
    return errores


def predecir_archivo(ruta_input, ruta_modelo='modelo_completo.pkl', ruta_output='predicciones_tiendas.csv'):
    if not os.path.exists(ruta_input):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_input}")

    df = pd.read_csv(ruta_input)
    errores = validar_csv(df)
    if errores:
        for e in errores:
            print("[ERROR]", e)
        return

    modelo_path = os.path.join(os.path.dirname(__file__), ruta_modelo)
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {modelo_path}")

    modelo = joblib.load(modelo_path)
    X = df[FEATURE_COLUMNS]
    predicciones = modelo.predict(X)
    df['vale_abrir_pred'] = predicciones
    df.to_csv(ruta_output, index=False)
    print(f"Predicciones guardadas en {ruta_output}")


if __name__ == "__main__":
    predecir_archivo("DIM_TIENDA_TEST.csv")
