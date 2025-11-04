# 🏃 Smart Peak Splits - Dashboard Streamlit

## Instalación

Primero, instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecuta la aplicación Streamlit con el siguiente comando:

```bash
streamlit run app_streamlit.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## 🎯 Funcionalidades

### 📁 Carga de Archivo
- Arrastra y suelta tu archivo GPX directamente
- Soporte para archivos .gpx

### 🚀 Predicción
- Genera predicciones automáticas con un clic
- Procesamiento rápido del archivo GPX

### 📊 Dashboard Interactivo

#### Métricas Principales
- ✓ Total de splits
- ✓ Ritmo promedio (min/km)
- ✓ Frecuencia cardíaca promedio (bpm)
- ✓ Elevación total (m)

#### Gráficos
1. **Ritmo por Split**: Línea con área sombreada mostrando la variación del ritmo
2. **Frecuencia Cardíaca**: Evolución de HR a lo largo de la carrera
3. **Elevación**: 
   - Ganancia de elevación por split
   - Elevación acumulada a lo largo de la carrera
4. **Tabla de Datos**: Vista completa de todas las predicciones

### 📥 Descarga de Datos
- Exporta todas las predicciones en formato CSV
- Nombre automático basado en el archivo GPX

### 📈 Resumen de Predicción
- Tiempo estimado de carrera
- Distancia total
- Evaluación de dificultad (Baja/Media/Alta)

## 📊 Estructura del Dashboard

```
┌─────────────────────────────────────────────────┐
│   🏃 Smart Peak Splits - Predictor              │
├─────────────────────────────────────────────────┤
│                                                  │
│  📁 Carga: [GPX File Uploader]                  │
│  🚀 Botón: [Generar Predicción]                 │
│                                                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  📊 Resumen General                             │
│  ┌─────────┬──────────┬──────────┬──────────┐  │
│  │ Splits  │ Ritmo    │ HR       │ Elevación│  │
│  └─────────┴──────────┴──────────┴──────────┘  │
│                                                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  📈 Gráficos (Tabs)                             │
│  [ Ritmo | HR | Elevación | Datos ]             │
│                                                  │
│  - Gráficos interactivos con matplotlib         │
│  - Tabla con predicciones completas             │
│  - Botón de descarga CSV                        │
│                                                  │
├─────────────────────────────────────────────────┤
│  🎯 Resumen Final                               │
│  ⏱️ Tiempo estimado | 📊 Distancia | 🏔️ Dificultad
└─────────────────────────────────────────────────┘
```

## 🖥️ Uso Típico

1. **Inicia la aplicación**:
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Carga tu GPX**: 
   - Arrastra el archivo o úsalo el selector
   - Espera confirmación de carga

3. **Genera predicciones**:
   - Haz clic en "🚀 Generar Predicción"
   - Espera el procesamiento

4. **Analiza resultados**:
   - Revisa métricas principales
   - Examina gráficos por pestañas
   - Consulta datos en la tabla

5. **Descarga datos**:
   - Haz clic en "⬇️ Descargar como CSV"
   - Se descargará con nombre automático

## 🔧 Requisitos

- Python 3.8+
- TensorFlow (modelo preentrenado)
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
- GPXpy

## 📝 Notas

- El modelo debe estar entrenado previamente
- Se busca en: `notebooks-research/models/smart_peak_model.keras`
- Los archivos escaladores deben estar disponibles
- La aplicación es completamente local, sin datos enviados a servidores

## 🎨 Personalización

Puedes modificar colores, estilos y disposición editando `app_streamlit.py`:
- Colores en `st.markdown()` con CSS
- Disposición de columnas con `st.columns()`
- Gráficos con `matplotlib` y `seaborn`

¡Disfruta de tu dashboard! 🚀
