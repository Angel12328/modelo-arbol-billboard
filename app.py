import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción Billboard Top 1",
    page_icon="🎵",
    layout="wide"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #B3B3B3;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
    }
    .top1 {
        background-color: #1db9542e;
        border: 2px solid #1DB954;
        color: #1DB954;
    }
    .not-top1 {
        background-color: #e221342e;
        border: 2px solid #E22134;
        color: #E22134;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎵 Billboard Top 1 Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ingresa las características de tu canción y averigua si llegará al puesto #1</div>', unsafe_allow_html=True)

# Cargar Modelo o Metadata
@st.cache_resource
def cargar_modelo():
    model_path = "importancia_variables.csv"  # Nombre del archivo pkl/csv proporcionado
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            return None
    else:
        st.warning(f"No se encontró el archivo del modelo `{model_path}`.")
        return None

model = cargar_modelo()

# Columnas esperadas por el modelo según la metadata
FEATURE_COLUMNS = [
    "durationSeg", "artist_type_Male", "artist_type_Mixed",
    "mood_Brooding", "mood_Cool", "mood_Defiant", "mood_Easygoing",
    "mood_Empowering", "mood_Energizing", "mood_Excited", "mood_Fiery",
    "mood_Gritty", "mood_Lively", "mood_Melancholy", "mood_Other",
    "mood_Peaceful", "mood_Romantic", "mood_Rowdy", "mood_Sensual",
    "mood_Sentimental", "mood_Sophisticated", "mood_Stirring", "mood_Upbeat",
    "mood_Urgent", "mood_Yearning", "tempo_Medium Tempo", "tempo_Slow Tempo",
    "genre_Electronica", "genre_Jazz", "genre_Other", "genre_Pop",
    "genre_Rock", "genre_Soundtrack", "genre_Traditional", "genre_Urban"
]

# Formulario de entrada
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Datos Generales")
    duracion_min = st.number_input("Duración (minutos)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
    duracion_seg = int(duracion_min * 60)
    st.caption(f"Equivalente a: **{duracion_seg} segundos**")

    artist_type = st.selectbox(
        "Tipo de Artista",
        options=["Femenino / Solista Femenino (por defecto)", "Masculino (Male)", "Mixto / Grupo (Mixed)"]
    )

    tempo = st.selectbox(
        "Tempo de la Canción",
        options=["Rápido (Fast Tempo - por defecto)", "Medio (Medium Tempo)", "Lento (Slow Tempo)"]
    )

with col2:
    st.subheader("🎨 Estilo y Actitud")
    
    mood_options = [
        "Brooding", "Cool", "Defiant", "Easygoing", "Empowering", 
        "Energizing", "Excited", "Fiery", "Gritty", "Lively", 
        "Melancholy", "Peaceful", "Romantic", "Rowdy", "Sensual", 
        "Sentimental", "Sophisticated", "Stirring", "Upbeat", 
        "Urgent", "Yearning", "Otro / Sin especificar"
    ]
    mood = st.selectbox("Mood / Estado de Ánimo", options=mood_options)

    genre_options = [
        "Electronica", "Jazz", "Pop", "Rock", 
        "Soundtrack", "Traditional", "Urban", "Otro"
    ]
    genre = st.selectbox("Género Musical", options=genre_options)

st.markdown("---")

# Preprocesamiento de datos
def construir_vector_caracteristicas():
    # Inicializar todas las columnas en 0
    input_data = {col: 0 for col in FEATURE_COLUMNS}
    
    # Asignar duración
    input_data["durationSeg"] = duracion_seg
    
    # Asignar tipo de artista (One-Hot Encoding)
    if artist_type == "Masculino (Male)":
        input_data["artist_type_Male"] = 1
    elif artist_type == "Mixto / Grupo (Mixed)":
        input_data["artist_type_Mixed"] = 1
        
    # Asignar tempo
    if tempo == "Medio (Medium Tempo)":
        input_data["tempo_Medium Tempo"] = 1
    elif tempo == "Lento (Slow Tempo)":
        input_data["tempo_Slow Tempo"] = 1
        
    # Asignar mood
    if mood != "Otro / Sin especificar":
        mood_key = f"mood_{mood}"
        if mood_key in input_data:
            input_data[mood_key] = 1
        else:
            input_data["mood_Other"] = 1
    else:
        input_data["mood_Other"] = 1

    # Asignar género
    genre_key = f"genre_{genre}"
    if genre_key in input_data:
        input_data[genre_key] = 1
    else:
        input_data["genre_Other"] = 1
        
    return pd.DataFrame([input_data])

df_input = construir_vector_caracteristicas()

# Botón para Predecir
if st.button("🚀 Evaluar Canción", use_container_width=True):
    if model is not None:
        try:
            prediccion = model.predict(df_input)[0]
            probabilidades = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else None

            if prediccion == 1:
                st.markdown(
                    '<div class="result-box top1">🌟 ¡Felicidades! La canción TIENE alto potencial de llegar al TOP 1 de Billboard.</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-box not-top1">⚠️ La canción NO cumple los patrones históricos para alcanzar el TOP 1.</div>', 
                    unsafe_allow_html=True
                )

            if probabilidades is not None:
                st.write("---")
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Probabilidad No Top 1", f"{probabilidades[0]*100:.2f}%")
                col_p2.metric("Probabilidad Top 1", f"{probabilidades[1]*100:.2f}%")

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
    else:
        st.error("El modelo no está cargado correctamente. Verifica que 'importancia_variables.csv' esté en la misma carpeta.")

# Mostrar variables codificadas para inspección
with st.expander("🔍 Ver vector de variables enviadas al modelo"):
    st.dataframe(df_input)