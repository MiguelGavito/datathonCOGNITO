import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def cargar_y_filtrar_tiendas(ruta_csv, ruta_shp_nl, ruta_shp_tam):
    df = pd.read_csv(ruta_csv)
    nuevo_leon = gpd.read_file(ruta_shp_nl)
    tamaulipas = gpd.read_file(ruta_shp_tam)

    geom_nl = nuevo_leon.geometry.unary_union
    geom_tam = tamaulipas.geometry.unary_union

    filas_validas = []
    for _, row in df.iterrows():
        lat, lon = row["LATITUD_NUM"], row["LONGITUD_NUM"]
        punto = Point(lon, lat)
        if geom_nl.contains(punto) or geom_tam.contains(punto):
            filas_validas.append(row)

    return pd.DataFrame(filas_validas)


def limpiar_columnas_categoricas(df):
    columnas = ['MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM', 'CAJONESESTACIONAMIENTO_NUM']
    for col in columnas:
        media = df[col][df[col] != 0].mean()
        df[col] = df[col].replace(0, media)
    return df


def calcular_tiendas_exitosas(venta, meta_venta, dim_tienda):
    ventas_completas = venta.merge(dim_tienda[['TIENDA_ID', 'ENTORNO_DES']], on='TIENDA_ID', how='left')
    ventas_completas = ventas_completas.merge(meta_venta, on='ENTORNO_DES', how='left')

    stats = ventas_completas.groupby('TIENDA_ID')['VENTA_TOTAL'].agg(['mean', 'std']).reset_index()
    stats.rename(columns={'mean': 'MEDIA', 'std': 'STD'}, inplace=True)
    ventas_completas = ventas_completas.merge(stats, on='TIENDA_ID', how='left')

    umbral_inferior = ventas_completas['MEDIA'] - 2 * ventas_completas['STD']
    ventas_completas = ventas_completas[ventas_completas['VENTA_TOTAL'] >= umbral_inferior]

    ventas_completas['ANIO'] = ventas_completas['MES_ID'].astype(str).str[:4].astype(int)
    ventas_completas['MES_EXITOSO'] = (ventas_completas['VENTA_TOTAL'] >= ventas_completas['Meta_venta']).astype(int)

    resumen = ventas_completas.groupby(['TIENDA_ID', 'ANIO']).agg(
        MESES_VALIDOS=('MES_ID', 'count'),
        MESES_EXITOSOS=('MES_EXITOSO', 'sum')
    ).reset_index()

    resumen['PORC_EXITOSO'] = resumen['MESES_EXITOSOS'] / resumen['MESES_VALIDOS']
    resumen['CUMPLE'] = resumen['PORC_EXITOSO'] >= 0.58

    tiendas_exitosas = (
        resumen[resumen['CUMPLE']]
        .groupby('TIENDA_ID')
        .size()
        .reset_index(name='Anios_OK')
    )

    return tiendas_exitosas[tiendas_exitosas['Anios_OK'] == 2]['TIENDA_ID']


def etiquetar_tiendas_exitosas(df_filtrado, tiendas_exitosas):
    df_filtrado['vale_abrir'] = df_filtrado['TIENDA_ID'].isin(tiendas_exitosas).astype(int)
    return df_filtrado