import pandas as pd
import joblib
import os

FEATURE_COLUMNS = [
    'PLAZA_CVE', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES',
    'MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM',
    'LATITUD_NUM', 'LONGITUD_NUM', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA'
]

ALL_COLUMNS = ['TIENDA_ID'] + FEATURE_COLUMNS + ['DATASET']

salida_csv = 'resultados_terminal.csv'
df_out = pd.DataFrame(columns=ALL_COLUMNS + ['vale_abrir_pred'])

modelo_path = os.path.join(os.path.dirname(__file__), 'modelo_completo.pkl')
if not os.path.exists(modelo_path):
    raise FileNotFoundError("No se encontró el modelo entrenado.")
modelo = joblib.load(modelo_path)

print("\n--- Ingreso de datos para predicción ---")
while True:
    entrada = {}
    entrada['TIENDA_ID'] = input("TIENDA_ID (opcional): ") or None
    for col in FEATURE_COLUMNS:
        entrada[col] = input(f"{col}: ")
    entrada['DATASET'] = input("DATASET: ")

    try:
        entrada_formato = {
            'TIENDA_ID': int(entrada['TIENDA_ID']) if entrada['TIENDA_ID'] else None,
            'PLAZA_CVE': str(entrada['PLAZA_CVE']),
            'NIVELSOCIOECONOMICO_DES': str(entrada['NIVELSOCIOECONOMICO_DES']),
            'ENTORNO_DES': str(entrada['ENTORNO_DES']),
            'MTS2VENTAS_NUM': float(entrada['MTS2VENTAS_NUM']),
            'PUERTASREFRIG_NUM': float(entrada['PUERTASREFRIG_NUM']),
            'CAJONESESTACIONAMIENTO_NUM': float(entrada['CAJONESESTACIONAMIENTO_NUM']),
            'LATITUD_NUM': float(entrada['LATITUD_NUM']),
            'LONGITUD_NUM': float(entrada['LONGITUD_NUM']),
            'SEGMENTO_MAESTRO_DESC': str(entrada['SEGMENTO_MAESTRO_DESC']),
            'LID_UBICACION_TIENDA': str(entrada['LID_UBICACION_TIENDA']),
            'DATASET': str(entrada['DATASET'])
        }

        df_tmp = pd.DataFrame([entrada_formato])
        pred = modelo.predict(df_tmp[FEATURE_COLUMNS])[0]
        entrada_formato['vale_abrir_pred'] = int(pred)
        print(f"\nResultado: {'Vale la pena abrir' if pred == 1 else 'No vale la pena abrir'}")

        df_out = pd.concat([df_out, pd.DataFrame([entrada_formato])], ignore_index=True)
        df_out.to_csv(salida_csv, index=False)
        print(f"Guardado en {salida_csv}\n")

    except Exception as e:
        print(f"[ERROR] Datos inválidos: {e}\n")

    continuar = input("¿Deseas ingresar otra tienda? (s/n): ")
    if continuar.lower() != 's':
        print("\nFinalizando ingreso de datos.")
        break