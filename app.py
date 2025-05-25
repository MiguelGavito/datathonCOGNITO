import streamlit as st
import pandas as pd
import joblib
import os
import pydeck as pdk
from shapely.geometry import Point
import geopandas as gpd
from filter import cargar_y_filtrar_tiendas, limpiar_columnas_categoricas
import tempfile

# Archivos de shapefiles para los estados
RUTA_SHP_NL = "2024_1_19_ENT.shp"
RUTA_SHP_TAM = "2024_1_28_ENT.shp"

# Configuración de la página
st.set_page_config(layout="wide", page_title="COGNITO Predict Tiendas OXXO")
st.title("📍 Predicción de Apertura de Tiendas OXXO")

# Ruta del modelo
modelo_path = "modelo_completo.pkl"
if not os.path.exists(modelo_path):
    st.error("No se encontró el modelo.")
    st.stop()

# Cargar modelo
modelo = joblib.load(modelo_path)

# Columnas esperadas por el modelo
FEATURE_COLUMNS = [
    'PLAZA_CVE', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES',
    'MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM',
    'LATITUD_NUM', 'LONGITUD_NUM', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA'
]
ALL_COLUMNS = ['TIENDA_ID'] + FEATURE_COLUMNS + ['DATASET']

# Carga de archivo CSV completo
uploaded_file = st.file_uploader("📁 Cargar CSV con tiendas para analizar", type="csv")

if uploaded_file:
  try:
      with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        ruta_temporal = tmp.name

      df_raw = pd.read_csv(ruta_temporal)
      df_filtrado = cargar_y_filtrar_tiendas(ruta_temporal, RUTA_SHP_NL, RUTA_SHP_TAM)
      print(2)
      if df_filtrado.empty:
          st.warning("El archivo no contiene tiendas dentro de Nuevo León o Tamaulipas.")
          st.stop()
      print(3)
      if not all(col in df_filtrado.columns for col in FEATURE_COLUMNS):
          st.error("❌ El archivo no tiene todas las columnas requeridas.")
          st.stop()
      print(4)
      df_filtrado = limpiar_columnas_categoricas(df_filtrado)
      df_filtrado['vale_abrir_pred'] = modelo.predict(df_filtrado[FEATURE_COLUMNS])
      print(5)
      st.subheader("📋 Resultados")
      st.dataframe(df_filtrado)
      print(6)
      col1, col2 = st.columns(2)
      col1.metric("✅ Tiendas que vale abrir", df_filtrado['vale_abrir_pred'].sum())
      col2.metric("❌ Tiendas que NO vale abrir", len(df_filtrado) - df_filtrado['vale_abrir_pred'].sum())
      print(7)
      st.subheader("🗺️ Mapa de tiendas recomendadas")
      df_mapa = df_filtrado.copy()
      df_mapa["color"] = df_mapa["vale_abrir_pred"].map({1: [0, 255, 0], 0: [255, 0, 0]})
      print(8)
      st.pydeck_chart(pdk.Deck(
          initial_view_state=pdk.ViewState(
              latitude=df_filtrado['LATITUD_NUM'].mean(),
              longitude=df_filtrado['LONGITUD_NUM'].mean(),
              zoom=10,
              pitch=0,
          ),
          layers=[
              pdk.Layer(
                  'ScatterplotLayer',
                  data=df_mapa,
                  get_position='[LONGITUD_NUM, LATITUD_NUM]',
                  get_fill_color='color',
                  get_radius=50,
                  pickable=True,
              )
          ],
          tooltip={"text": "Tienda ID: {TIENDA_ID}\nVale abrir: {vale_abrir_pred}"}
      ))

      st.download_button(
          label="📥 Descargar CSV con predicción",
          data=df_filtrado.to_csv(index=False).encode('utf-8'),
          file_name="predicciones_tiendas.csv",
          mime='text/csv'
      )
  except Exception as e:
      st.error(f"[ERROR] No se pudo procesar el archivo: {e}")

# Separador
st.markdown("---")
st.subheader("🧮 Predicción individual de tienda")


with st.form("input_form"):
    tienda_id = st.text_input("TIENDA_ID (opcional):")
    plaza_cve = st.text_input("PLAZA_CVE:")
    nivel_socio = st.text_input("NIVELSOCIOECONOMICO_DES:")
    entorno = st.text_input("ENTORNO_DES:")
    mts2 = st.number_input("MTS2VENTAS_NUM:", format="%.2f")
    puertas = st.number_input("PUERTASREFRIG_NUM:", format="%.0f")
    cajones = st.number_input("CAJONESESTACIONAMIENTO_NUM:", format="%.0f")
    latitud = st.number_input("LATITUD_NUM:", format="%.5f")
    longitud = st.number_input("LONGITUD_NUM:", format="%.5f")
    segmento = st.text_input("SEGMENTO_MAESTRO_DESC:")
    lid_ubicacion = st.text_input("LID_UBICACION_TIENDA:")
    dataset = st.text_input("DATASET:")

    submitted = st.form_submit_button("Predecir")

if submitted:
    try:
        # Validación geográfica individual
        nl_geom = gpd.read_file(RUTA_SHP_NL).geometry.unary_union
        tam_geom = gpd.read_file(RUTA_SHP_TAM).geometry.unary_union
        punto = Point(float(longitud), float(latitud))

        if not (nl_geom.contains(punto) or tam_geom.contains(punto)):
            st.warning("⚠️ La ubicación no pertenece a Nuevo León ni Tamaulipas.")
            st.stop()

        entrada_formato = {
            'TIENDA_ID': int(tienda_id) if tienda_id else 0,
            'PLAZA_CVE': plaza_cve,
            'NIVELSOCIOECONOMICO_DES': nivel_socio,
            'ENTORNO_DES': entorno,
            'MTS2VENTAS_NUM': float(mts2),
            'PUERTASREFRIG_NUM': float(puertas),
            'CAJONESESTACIONAMIENTO_NUM': float(cajones),
            'LATITUD_NUM': float(latitud),
            'LONGITUD_NUM': float(longitud),
            'SEGMENTO_MAESTRO_DESC': segmento,
            'LID_UBICACION_TIENDA': lid_ubicacion,
            'DATASET': dataset
        }

        df_individual = pd.DataFrame([entrada_formato])[ALL_COLUMNS]
        df_individual = limpiar_columnas_categoricas(df_individual)

        pred = modelo.predict(df_individual[FEATURE_COLUMNS])[0]
        df_individual['vale_abrir_pred'] = int(pred)

        if pred == 1:
            st.success("✅ ¡Vale la pena abrir esta tienda!")
        else:
            st.error("❌ No es rentable abrir esta tienda.")

        salida_csv = 'resultados_streamlit.csv'
        if os.path.exists(salida_csv):
            df_out = pd.read_csv(salida_csv)
            df_out = pd.concat([df_out, df_individual], ignore_index=True)
        else:
            df_out = df_individual
        df_out.to_csv(salida_csv, index=False)

    except Exception as e:
        st.error(f"[ERROR] Datos inválidos: {e}")