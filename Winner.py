import streamlit as st
import pandas as pd
import requests
import time
import os
import numpy as np

# Librerías para Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Loto Analytics & AI", layout="wide")

st.title("🎱 Análisis y Predicción de Loto (Extracción API + AI)")
st.markdown("""
**Autor:** José Luis  
**Objetivo:** Extracción eficiente de datos vía peticiones HTTP (Requests) a la API oculta del sitio web, evitando navegadores pesados. Ideal para despliegues Serverless en Streamlit Cloud.
""")

# ==========================================
# 1. MÓDULO DE SCRAPING (API REQUESTS)
# ==========================================
def extraer_datos_api(url_api_endpoint, max_pages=10):
    """
    Realiza peticiones HTTP directas al endpoint que devuelve el JSON.
    NOTA METODOLÓGICA: Reemplazar 'url_api_endpoint' por la URL real extraída 
    desde la pestaña "Red" (F12) del navegador.
    """
    datos_totales = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Headers para simular ser un navegador real y evitar bloqueos (WAF)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "es-CL,es;q=0.9",
        "Referer": "https://www.polla.cl/"
    }

    try:
        status_text.text("Iniciando conexión a la API de datos...")
        
        for page in range(1, max_pages + 1):
            # Construcción dinámica del endpoint paginado
            target_url = f"{url_api_endpoint}?page={page}" 
            
            response = requests.get(target_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # === ADAPTACIÓN JSON ===
                    # Dependiendo de la web, estas claves ('results', 'drawNumber', 'numbers')
                    # deben ajustarse a la estructura real del JSON.
                    resultados_pagina = data.get('results', [])
                    
                    if not resultados_pagina:
                        status_text.text("No se encontraron más datos en la API. Fin de la extracción.")
                        break

                    for item in resultados_pagina:
                        sorteo_int = int(item.get('drawNumber', 0))
                        numeros_raw = item.get('numbers', '')
                        
                        if sorteo_int != 0 and not any(d['Numero de sorteo'] == sorteo_int for d in datos_totales):
                            datos_totales.append({
                                'Numero de sorteo': sorteo_int,
                                'Numeros Ganadores': numeros_raw
                            })
                            
                except ValueError:
                    st.error("La respuesta no es un JSON válido. Verifica el endpoint.")
                    break
            else:
                st.warning(f"Error HTTP {response.status_code} al consultar la página {page}.")
                break
                
            # Retardo para no saturar el servidor objetivo (Cortesía HTTP)
            time.sleep(1.5)
            progress_bar.progress(min(page / max_pages, 1.0))
            status_text.text(f"Scrapeando página {page}. Registros encontrados: {len(datos_totales)}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error crítico de red: {e}")
        
    return pd.DataFrame(datos_totales)

# ==========================================
# 2. MÓDULO DE ETL (TRANSFORMACIÓN)
# ==========================================
def procesar_datos(df):
    """
    Transforma la columna de string de números en columnas separadas.
    """
    if df.empty:
        return df

    def split_numbers(row):
        raw = str(row['Numeros Ganadores'])
        raw = raw.replace('|', ',') 
        parts = [p.strip() for p in raw.split(',')]
        
        numeros = [int(x) for x in parts if x.isdigit()]
        
        if len(numeros) >= 7:
            return pd.Series(numeros[:7])
        else:
            return pd.Series([0]*7)

    new_cols = df.apply(split_numbers, axis=1)
    new_cols.columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'Comodin']
    
    df_final = pd.concat([df['Numero de sorteo'], new_cols], axis=1)
    df_final = df_final.sort_values(by='Numero de sorteo', ascending=False)
    
    return df_final

# ==========================================
# 3. MÓDULO DE DATA SCIENCE (PREDICCIÓN)
# ==========================================
def crear_dataset_supervisado(data, window_size=5):
    X, y = [], []
    valores = data[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
    valores = valores[::-1] # Orden cronológico para aprendizaje correcto
    
    for i in range(len(valores) - window_size):
        X.append(valores[i:i+window_size].flatten())
        y.append(valores[i+window_size])
        
    return np.array(X), np.array(y)

def entrenar_y_predecir(df, modelos_seleccionados, n_predicciones=3):
    window_size = 5
    X, y = crear_dataset_supervisado(df, window_size)
    
    if len(X) < 10:
        st.warning("Datos insuficientes para ML.")
        return {}

    ultimos_datos_reales = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].head(window_size).values[::-1]
    input_inicial = ultimos_datos_reales.flatten().reshape(1, -1)
    
    resultados = {}

    for nombre, modelo_base in modelos_seleccionados.items():
        model = MultiOutputRegressor(modelo_base)
        model.fit(X, y)
        
        predicciones_serie = []
        current_input = input_inicial.copy()
        
        for _ in range(n_predicciones):
            pred = model.predict(current_input)
            pred = np.round(pred).astype(int)
            pred = np.clip(pred, 1, 41)
            pred[0].sort()
            
            predicciones_serie.append(pred[0])
            
            nuevo_input = np.append(current_input[0][6:], pred[0]).reshape(1, -1)
            current_input = nuevo_input
            
        resultados[nombre] = predicciones_serie
        
    return resultados

# ==========================================
# INTERFAZ DE USUARIO (MAIN)
# ==========================================
def main():
    st.sidebar.header("Panel de Control")
    
    modo = st.sidebar.selectbox("Modo de Operación", ["Cargar CSV Existente", "Nuevo Scraping (API Requests)", "Datos de Prueba"])
    
    if modo == "Nuevo Scraping (API Requests)":
        st.sidebar.info("Modo ultraligero sin navegador. Extrae vía API RESTful.")
        
        url_target = st.sidebar.text_input("Endpoint API JSON", "https://api.ejemplo.cl/resultados")
        max_pags = st.sidebar.number_input("Máximas Páginas a recorrer", value=5, min_value=1, max_value=1000)
        
        if st.sidebar.button("Extraer Datos"):
            with st.spinner("Conectando con la API y extrayendo lotes..."):
                df_raw = extraer_datos_api(url_target, max_pages=max_pags)
                
            if not df_raw.empty:
                df_clean = procesar_datos(df_raw)
                df_clean.to_csv("Numeros.csv", index=False)
                st.sidebar.success(f"Extracción finalizada. {len(df_clean)} registros guardados.")
            else:
                st.sidebar.error("No se extrajeron datos. Revisa la URL del Endpoint en la consola F12.")

    elif modo == "Datos de Prueba":
        if st.sidebar.button("Generar Mock Data"):
            data_mock = []
            for i in range(5355, 5300, -1):
                nums = np.sort(np.random.choice(range(1, 42), 6, replace=False))
                comodin = np.random.randint(1, 42)
                row = [i] + list(nums) + [comodin]
                data_mock.append(row)
            df_mock = pd.DataFrame(data_mock, columns=['Numero de sorteo', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'Comodin'])
            df_mock.to_csv("Numeros.csv", index=False)
            st.sidebar.success("Datos simulados creados.")

    # --- Visualización ---
    if os.path.exists("Numeros.csv"):
        st.subheader("1. Datos Históricos (ETL)")
        df = pd.read_csv("Numeros.csv")
        st.dataframe(df.head(10))
        st.caption(f"Total registros: {len(df)}")
        
        st.subheader("2. Predicciones con IA")
        st.markdown("Generando 3 series probables basadas en patrones históricos.")
        
        modelos = {
            "Random Forest (Ensamble)": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting (Boosting)": GradientBoostingRegressor(n_estimators=200, random_state=42),
            "K-Nearest Neighbors (Distancia)": KNeighborsRegressor(n_neighbors=7)
        }
        
        if st.button("Calcular Probabilidades"):
            with st.spinner("Los modelos están analizando las series temporales..."):
                preds = entrenar_y_predecir(df, modelos)
            
            col1, col2, col3 = st.columns(3)
            
            def mostrar_resultados(col, nombre, series):
                with col:
                    st.markdown(f"### {nombre}")
                    for i, serie in enumerate(series):
                        st.info(f"**Jugada {i+1}:** {serie}")

            items = list(preds.items())
            if len(items) >= 3:
                mostrar_resultados(col1, items[0][0], items[0][1])
                mostrar_resultados(col2, items[1][0], items[1][1])
                mostrar_resultados(col3, items[2][0], items[2][1])
            
            st.warning("Recuerda: El Loto es un juego de azar. Estas predicciones son exploratorias.")

if __name__ == "__main__":
    main()
