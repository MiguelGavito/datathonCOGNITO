import streamlit as st
import pandas as pd
import joblib
import os
import pydeck as pdk

# Configuración inicial
st.set_page_config(layout="wide", page_title="Dashboard Tiendas OXXO")
st.title("📍 Predicción de Apertura de Tiendas OXXO")

# Carga del modelo
modelo_path = "modelo_completo.pkl"
if not os.path.exists(modelo_path):
    st.error("❌ No se encontró el modelo.")
    st.stop()

modelo = joblib.load(modelo_path)

# Columnas del modelo
FEATURE_COLUMNS = [
    'PLAZA_CVE', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES',
    'MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM',
    'LATITUD_NUM', 'LONGITUD_NUM', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA'
]

# Carga del archivo
uploaded_file = st.file_uploader("📁 Cargar CSV con tiendas para analizar", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validación rápida de columnas
    if not all(col in df.columns for col in FEATURE_COLUMNS):
        st.error("❌ El archivo no tiene todas las columnas requeridas.")
        st.stop()

    # Predicción
    df['vale_abrir_pred'] = modelo.predict(df[FEATURE_COLUMNS])

    # Mostrar tabla
    st.subheader("📋 Resultados")
    st.dataframe(df)

    # Visualización de métricas
    col1, col2 = st.columns(2)
    con_si = df['vale_abrir_pred'].sum()
    con_no = len(df) - con_si

    col1.metric("✅ Tiendas que vale abrir", con_si)
    col2.metric("❌ Tiendas que NO vale abrir", con_no)

    # Visualización en mapa
    st.subheader("🗺️ Mapa de tiendas recomendadas")
    df_mapa = df.copy()
    df_mapa["color"] = df_mapa["vale_abrir_pred"].map({1: [0, 255, 0], 0: [255, 0, 0]})

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=df['LATITUD_NUM'].mean(),
            longitude=df['LONGITUD_NUM'].mean(),
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

    # Descarga del resultado
    st.download_button(
        label="📥 Descargar CSV con predicción",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="predicciones_tiendas.csv",
        mime='text/csv'
    )
