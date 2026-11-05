import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Billboard Top 1 Predictor",
    page_icon="🎵",
    layout="wide"
)

# Cargar el modelo entrenado y la importancia de variables
@st.cache_resource
def cargar_modelo():
    modelo_path = "arbol_decision_billboard.pkl"
    if not os.path.exists(modelo_path):
        st.error(f"⚠️ No se encontró el archivo '{modelo_path}' en el repositorio. Asegúrate de subirlo a la raíz del proyecto.")
        return None
    return joblib.load(modelo_path)

@st.cache_data
def cargar_importancia():
    csv_path = "importancia_variables.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

model = cargar_modelo()
df_importancia = cargar_importancia()

# Título y Encabezado
st.title("🎵 Billboard Top 1 Predictor")
st.write("Ingresa las características de tu canción y averigua si llegará al puesto #1 en los Billboard.")

st.divider()

if model is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Datos Generales")
        
        # 1. Duración
        duracion_min = st.number_input(
            "Duración (minutos):",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.10
        )
        duracion_seg = duracion_min * 60
        st.caption(f"Equivalente a: **{int(duracion_seg)} segundos**")

        # 2. Tipo de Artista
        opciones_artista = [
            "Femenino / Solista Femenino (por defecto)",
            "Masculino (Male)",
            "Grupo / Mixto (Mixed)"
        ]
        tipo_artista = st.selectbox("Tipo de Artista:", opciones_artista)

        # 3. Tempo de la Canción
        opciones_tempo = [
            "Rápido (Fast Tempo - por defecto)",
            "Medio (Medium Tempo)",
            "Lento (Slow Tempo)"
        ]
        tempo = st.selectbox("Tempo de la Canción:", opciones_tempo)

    with col2:
        st.subheader("🎨 Estilo y Actitud")
        
        # 4. Mood / Estado de Ánimo
        lista_moods = [
            "Brooding", "Cool", "Defiant", "Easygoing", "Empowering", 
            "Energizing", "Excited", "Fiery", "Gritty", "Lively", 
            "Melancholy", "Other", "Peaceful", "Romantic", "Rowdy", 
            "Sensual", "Sentimental", "Sophisticated", "Stirring", 
            "Upbeat", "Urgent", "Yearning"
        ]
        mood_seleccionado = st.selectbox("Mood / Estado de Ánimo:", lista_moods)

        # 5. Género Musical
        lista_generos = [
            "Electronica", "Jazz", "Other", "Pop", "Rock", 
            "Soundtrack", "Traditional", "Urban"
        ]
        genero_seleccionado = st.selectbox("Género Musical:", lista_generos)

    st.divider()

    # Botón de Predicción
    if st.button("🚀 Evaluar Canción", use_container_width=True):
        # Columnas exactas que espera el Árbol de Decisión
        columnas_modelo = [
            'durationSeg', 'artist_type_Male', 'artist_type_Mixed',
            'mood_Brooding', 'mood_Cool', 'mood_Defiant', 'mood_Easygoing',
            'mood_Empowering', 'mood_Energizing', 'mood_Excited', 'mood_Fiery',
            'mood_Gritty', 'mood_Lively', 'mood_Melancholy', 'mood_Other',
            'mood_Peaceful', 'mood_Romantic', 'mood_Rowdy', 'mood_Sensual',
            'mood_Sentimental', 'mood_Sophisticated', 'mood_Stirring',
            'mood_Upbeat', 'mood_Urgent', 'mood_Yearning',
            'tempo_Medium Tempo', 'tempo_Slow Tempo',
            'genre_Electronica', 'genre_Jazz', 'genre_Other', 'genre_Pop',
            'genre_Rock', 'genre_Soundtrack', 'genre_Traditional', 'genre_Urban'
        ]

        # Inicializar diccionario con ceros
        datos_entrada = {col: 0 for col in columnas_modelo}

        # Asignar duración
        datos_entrada['durationSeg'] = duracion_seg

        # Asignar Tipo de Artista (One-Hot Encoding)
        if "Male" in tipo_artista:
            datos_entrada['artist_type_Male'] = 1
        elif "Mixed" in tipo_artista:
            datos_entrada['artist_type_Mixed'] = 1

        # Asignar Tempo (One-Hot Encoding)
        if "Medium" in tempo:
            datos_entrada['tempo_Medium Tempo'] = 1
        elif "Slow" in tempo:
            datos_entrada['tempo_Slow Tempo'] = 1

        # Asignar Mood (One-Hot Encoding)
        col_mood = f"mood_{mood_seleccionado}"
        if col_mood in datos_entrada:
            datos_entrada[col_mood] = 1

        # Asignar Género (One-Hot Encoding)
        col_genero = f"genre_{genero_seleccionado}"
        if col_genero in datos_entrada:
            datos_entrada[col_genero] = 1

        # Convertir a DataFrame
        df_input = pd.DataFrame([datos_entrada])

        # Realizar Predicción
        prediccion = model.predict(df_input)[0]
        
        # Obtener probabilidades si el modelo las soporta
        probabilidades = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else None

        st.subheader("📌 Resultado de la Predicción")
        if prediccion == 1:
            st.success("🎉 **¡ÉXITO ASEGURADO!** El modelo predice que esta canción **llegará al Top 1 de Billboard**.")
            if probabilidades is not None:
                st.metric(label="Probabilidad calculada", value=f"{probabilidades[1]*100:.1f}%")
        else:
            st.warning("❌ **NO ALCANZARÁ EL TOP 1.** Según el modelo, esta canción no tiene las características típicas del puesto #1.")
            if probabilidades is not None:
                st.metric(label="Probabilidad de Top 1", value=f"{probabilidades[1]*100:.1f}%")

    # Mostrar la Tabla de Importancia de Variables si existe el CSV
    if df_importancia is not None:
        with st.expander("📈 Ver la Importancia de Variables del Modelo"):
            st.dataframe(df_importancia, use_container_width=True)
