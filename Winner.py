import streamlit as st
import pandas as pd
import time
import os
import numpy as np

# Librerías para Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Librerías para Scraping Avanzado (Selenium)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Loto Analytics & AI", layout="wide")

st.title("🎱 Análisis y Predicción de Loto (Selenium + AI)")
st.markdown("""
**Autor:** José Luis  
**Objetivo:** Scraping de SPA (Single Page Application) simulando interacción humana, ETL y predicción.
""")

# ==========================================
# 1. MÓDULO DE SCRAPING (SELENIUM)
# ==========================================
def iniciar_driver():
    """
    Configura e inicia una instancia de Chrome Headless.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ejecutar sin ventana gráfica para velocidad
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Instalación automática del driver compatible
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrapear_con_selenium(url_inicial, stop_at_sorteo_1=True, max_pages=10):
    """
    Navega a la URL fija y hace clic en 'Siguiente página' extrayendo datos del DOM renderizado.
    """
    datos_totales = []
    driver = iniciar_driver()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"Navegando a {url_inicial}...")
        driver.get(url_inicial)
        
        # Esperar a que cargue el primer contenedor de sorteo (Knockout.js tarda un poco)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-bind*='drawNumber']"))
        )

        page_count = 0
        
        while page_count < max_pages:
            # 1. Extraer datos de la página actual
            # Buscamos todos los contenedores de números y sorteos visibles en la pantalla actual
            # Nota: La estructura suele ser una lista de sorteos por página.
            
            # Buscamos los elementos usando los data-bind específicos que indicaste
            elementos_sorteo = driver.find_elements(By.CSS_SELECTOR, "div[data-bind*='text: drawNumber']")
            elementos_numeros = driver.find_elements(By.CSS_SELECTOR, "div[data-bind*='getResultNumbersString']")
            
            # Iteramos sobre los elementos encontrados en la vista actual
            # Asumimos que están en orden (el primer sorteo corresponde a los primeros números)
            for i in range(len(elementos_sorteo)):
                try:
                    sorteo_id = elementos_sorteo[i].text.strip()
                    numeros_raw = elementos_numeros[i].text.strip()
                    
                    if sorteo_id and numeros_raw:
                        # Convertir a int para verificar si llegamos al 1
                        sorteo_int = int(sorteo_id)
                        
                        # Evitar duplicados si la paginación solapa
                        if not any(d['Numero de sorteo'] == sorteo_int for d in datos_totales):
                            datos_totales.append({
                                'Numero de sorteo': sorteo_int,
                                'Numeros Ganadores': numeros_raw
                            })
                        
                        if stop_at_sorteo_1 and sorteo_int == 1:
                            st.success("¡Se alcanzó el Sorteo #1!")
                            return pd.DataFrame(datos_totales)
                except Exception as e:
                    continue

            # Actualizar progreso visual
            page_count += 1
            progress_bar.progress(min(page_count / max_pages, 1.0))
            status_text.text(f"Scrapeando página {page_count}. Registros encontrados: {len(datos_totales)}")
            
            # 2. Navegación (Paginación)
            try:
                # Buscamos el botón 'Siguiente página' según tu selector
                # El selector CSS busca un <a> que contenga el evento click goToNextPage
                next_btn = driver.find_element(By.CSS_SELECTOR, "a[data-bind*='click:  goToNextPage']")
                
                # Verificamos si está inactivo (clase 'inactive')
                classes = next_btn.get_attribute("class")
                if "inactive" in classes:
                    status_text.text("Se alcanzó la última página (botón inactivo).")
                    break
                
                # Hacemos click
                driver.execute_script("arguments[0].click();", next_btn)
                
                # Esperamos brevemente a que Knockout actualice el DOM
                time.sleep(1) 
                
            except Exception as e:
                status_text.warning(f"No se pudo avanzar más o error en paginación: {e}")
                break

    except Exception as e:
        st.error(f"Error crítico en Selenium: {e}")
    finally:
        driver.quit() # Importante: cerrar el navegador para liberar RAM
        
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
        # Formato: "11, 12, 13, 14, 16, 39 | 2"
        raw = row['Numeros Ganadores']
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
    # Aseguramos orden descendente por Sorteo
    df_final = df_final.sort_values(by='Numero de sorteo', ascending=False)
    
    return df_final

# ==========================================
# 3. MÓDULO DE DATA SCIENCE (PREDICCIÓN)
# ==========================================
def crear_dataset_supervisado(data, window_size=5):
    """
    Crea ventanas de tiempo para entrenamiento.
    """
    X, y = [], []
    # Seleccionamos solo las bolas principales (N1 a N6)
    valores = data[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
    
    # Invertimos para que el orden sea cronológico (Sorteo 1 -> Sorteo 5000)
    # Esto es crucial para que el modelo aprenda la secuencia histórica real.
    valores = valores[::-1] 
    
    for i in range(len(valores) - window_size):
        # Input: Ventana de 'window_size' sorteos previos
        X.append(valores[i:i+window_size].flatten())
        # Target: El siguiente sorteo
        y.append(valores[i+window_size])
        
    return np.array(X), np.array(y)

def entrenar_y_predecir(df, modelos_seleccionados, n_predicciones=3):
    window_size = 5
    X, y = crear_dataset_supervisado(df, window_size)
    
    if len(X) < 10:
        st.warning("Datos insuficientes para ML.")
        return {}

    # Tomamos los últimos 'window_size' registros REALES para empezar a predecir el futuro
    ultimos_datos_reales = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].head(window_size).values[::-1]
    input_inicial = ultimos_datos_reales.flatten().reshape(1, -1)
    
    resultados = {}

    for nombre, modelo_base in modelos_seleccionados.items():
        # MultiOutputRegressor permite predecir 6 variables al mismo tiempo
        model = MultiOutputRegressor(modelo_base)
        model.fit(X, y)
        
        predicciones_serie = []
        current_input = input_inicial.copy()
        
        for _ in range(n_predicciones):
            pred = model.predict(current_input)
            pred = np.round(pred).astype(int)
            pred = np.clip(pred, 1, 41) # Restricción de dominio del Loto
            
            # Ordenamos los números para presentación (como en el cartón)
            pred[0].sort()
            
            predicciones_serie.append(pred[0])
            
            # Actualizamos la ventana deslizante:
            # Eliminamos el sorteo más antiguo (primeros 6 elementos) y añadimos la nueva predicción al final
            nuevo_input = np.append(current_input[0][6:], pred[0]).reshape(1, -1)
            current_input = nuevo_input
            
        resultados[nombre] = predicciones_serie
        
    return resultados

# ==========================================
# INTERFAZ DE USUARIO (MAIN)
# ==========================================
def main():
    st.sidebar.header("Panel de Control")
    
    modo = st.sidebar.selectbox("Modo de Operación", ["Cargar CSV Existente", "Nuevo Scraping (Selenium)", "Datos de Prueba"])
    
    if modo == "Nuevo Scraping (Selenium)":
        st.sidebar.info("Requiere Chrome instalado en el servidor/local.")
        url_target = st.sidebar.text_input("URL Objetivo", "https://www.polla.cl/es/view/resultados/5271")
        max_pags = st.sidebar.number_input("Máximas Páginas a recorrer", value=5, min_value=1, max_value=1000)
        
        if st.sidebar.button("Iniciar Robot"):
            with st.spinner("Iniciando navegador virtual... esto puede tardar unos segundos."):
                df_raw = scrapear_con_selenium(url_target, max_pages=max_pags)
                
            if not df_raw.empty:
                df_clean = procesar_datos(df_raw)
                df_clean.to_csv("Numeros.csv", index=False)
                st.sidebar.success(f"Scraping finalizado. {len(df_clean)} registros guardados.")
            else:
                st.sidebar.error("No se extrajeron datos. Revisa la consola o selectores.")

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
            
            st.warning("Recuerda: El Loto es un juego de azar con eventos independientes. Estas predicciones son ejercicios matemáticos de búsqueda de patrones, no garantía de ganancia, jajajaj.")

if __name__ == "__main__":
    main()