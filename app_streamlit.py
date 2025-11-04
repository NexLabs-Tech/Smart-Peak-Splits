import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.predict import predict_race, extract_gpx_info
import os
import tempfile

# Configurar página
st.set_page_config(
    page_title="Smart Peak Splits - Predictor",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .title-main {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="title-main">🏃 Smart Peak Splits - Predictor de Carreras</div>', unsafe_allow_html=True)
st.markdown("### Dashboard de análisis y predicción de rendimiento en carreras")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("Carga tu archivo GPX y obtén predicciones de ritmo y frecuencia cardíaca")

# Área de carga de archivo
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Cargar archivo GPX")
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona un archivo GPX",
        type="gpx",
        help="Formato: .gpx"
    )

if uploaded_file is not None:
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Mostrar información del archivo
        st.success(f"✓ Archivo cargado: {uploaded_file.name}")
        
        # Botón para hacer predicción
        if st.button("🚀 Generar Predicción", use_container_width=True, type="primary"):
            with st.spinner("Procesando archivo GPX y generando predicciones..."):
                try:
                    # Hacer predicción
                    predictions_df = predict_race(tmp_path)
                    
                    if predictions_df is not None:
                        st.success("✓ Predicción completada exitosamente")
                        
                        # Almacenar en sesión
                        st.session_state.predictions = predictions_df
                        st.session_state.file_name = uploaded_file.name
                    else:
                        st.error("❌ Error al generar predicciones")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Mostrar predicciones si existen
        if 'predictions' in st.session_state:
            predictions_df = st.session_state.predictions
            
            st.markdown("---")
            st.subheader("📊 Resumen General")
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Splits",
                    f"{len(predictions_df)}",
                    help="Número de segmentos de 1km"
                )
            
            with col2:
                avg_pace = predictions_df['AVG_pace'].mean()
                st.metric(
                    "Ritmo Promedio",
                    f"{avg_pace:.2f} min/km",
                    help="Ritmo promedio en minutos por kilómetro"
                )
            
            with col3:
                avg_hr = predictions_df['avg_hr'].mean()
                st.metric(
                    "Frecuencia Cardíaca",
                    f"{avg_hr:.0f} bpm",
                    help="Frecuencia cardíaca promedio"
                )
            
            with col4:
                total_elevation = predictions_df['elevation_gain'].sum()
                st.metric(
                    "Elevación Total",
                    f"{total_elevation:.0f}m",
                    help="Ganancia de elevación acumulada"
                )
            
            st.markdown("---")
            st.subheader("📈 Gráficos de Análisis")
            
            # Crear tabs para diferentes vistas
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Mapa & Perfil", "Ritmo", "Frecuencia Cardíaca", "Elevación", "Datos"])
            
            # Tab 1: Mapa y Perfil
            with tab1:
                try:
                    # Extraer información del GPX
                    gpx_info = extract_gpx_info(tmp_path)
                    df_points = gpx_info['points']
                    
                    col_map, col_profile = st.columns([1, 1])
                    
                    # Mapa interactivo con Plotly
                    with col_map:
                        st.subheader("🗺️ Mapa de la Ruta")
                        fig_map = go.Figure()
                        
                        fig_map.add_trace(go.Scattergeo(
                            lon=df_points['longitude'],
                            lat=df_points['latitude'],
                            mode='lines',
                            name='Ruta',
                            line=dict(color='#1f77b4', width=4),
                            hovertemplate='<b>Coordenadas</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
                        ))
                        
                        # Punto inicial (verde)
                        fig_map.add_trace(go.Scattergeo(
                            lon=[df_points['longitude'].iloc[0]],
                            lat=[df_points['latitude'].iloc[0]],
                            mode='markers',
                            name='Inicio',
                            marker=dict(size=12, color='green', symbol='circle'),
                            hovertemplate='<b>Inicio</b><extra></extra>'
                        ))
                        
                        # Punto final (rojo)
                        fig_map.add_trace(go.Scattergeo(
                            lon=[df_points['longitude'].iloc[-1]],
                            lat=[df_points['latitude'].iloc[-1]],
                            mode='markers',
                            name='Final',
                            marker=dict(size=12, color='red', symbol='circle'),
                            hovertemplate='<b>Final</b><extra></extra>'
                        ))
                        
                        fig_map.update_geos(
                            projection_type='mercator',
                            showland=True,
                            landcolor='rgb(243, 243, 243)',
                            showocean=True,
                            oceancolor='rgb(204, 229, 255)',
                            showcountries=True,
                            countrycolor='rgb(204, 204, 204)'
                        )
                        
                        fig_map.update_layout(
                            title='Ruta GPS',
                            height=600,
                            margin=dict(r=0, t=30, l=0, b=0),
                            showlegend=True,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Perfil de elevación
                    with col_profile:
                        st.subheader("📊 Perfil de Elevación")
                        fig_profile = go.Figure()
                        
                        fig_profile.add_trace(go.Scatter(
                            x=df_points['cumulative_distance'],
                            y=df_points['elevation'],
                            fill='tozeroy',
                            name='Elevación',
                            line=dict(color='#9467bd', width=3),
                            fillcolor='rgba(148, 103, 189, 0.3)',
                            hovertemplate='<b>Distancia: %{x:.2f} km</b><br>Elevación: %{y:.0f}m<extra></extra>'
                        ))
                        
                        fig_profile.update_layout(
                            title='Perfil de Elevación',
                            xaxis_title='Distancia (km)',
                            yaxis_title='Elevación (m)',
                            height=600,
                            hovermode='x unified',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#333'),
                            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        )
                        
                        st.plotly_chart(fig_profile, use_container_width=True)
                    
                    # Información del GPX
                    st.markdown("---")
                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    
                    with info_col1:
                        st.metric(
                            "Distancia Total",
                            f"{gpx_info['total_distance']:.2f} km"
                        )
                    
                    with info_col2:
                        st.metric(
                            "Elevación Máxima",
                            f"{gpx_info['max_elevation']:.0f} m"
                        )
                    
                    with info_col3:
                        st.metric(
                            "Elevación Mínima",
                            f"{gpx_info['min_elevation']:.0f} m"
                        )
                    
                    with info_col4:
                        st.metric(
                            "Elevación Promedio",
                            f"{gpx_info['avg_elevation']:.0f} m"
                        )
                
                except Exception as e:
                    st.error(f"Error al procesar el mapa: {str(e)}")
            
            
            # Tab 2: Frecuencia Cardíaca
            with tab2:
                fig_pace = go.Figure()
                
                fig_pace.add_trace(go.Scatter(
                    x=predictions_df['split_km'],
                    y=predictions_df['AVG_pace'],
                    mode='lines+markers',
                    name='Ritmo',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.3)',
                    hovertemplate='<b>Split %{x}</b><br>Ritmo: %{y:.2f} min/km<extra></extra>'
                ))
                
                fig_pace.update_layout(
                    title='Predicción de Ritmo por Split',
                    xaxis_title='Split (km)',
                    yaxis_title='Ritmo (min/km)',
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    font=dict(color='#333'),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                )
                
                st.plotly_chart(fig_pace, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"⏱️ Ritmo Máximo: {predictions_df['AVG_pace'].max():.2f} min/km")
                with col2:
                    st.info(f"⏱️ Ritmo Mínimo: {predictions_df['AVG_pace'].min():.2f} min/km")
            
            # Tab 3: Frecuencia Cardíaca
            with tab3:
                fig_hr = go.Figure()
                
                fig_hr.add_trace(go.Scatter(
                    x=predictions_df['split_km'],
                    y=predictions_df['avg_hr'],
                    mode='lines+markers',
                    name='Frecuencia Cardíaca',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(255, 127, 14, 0.3)',
                    hovertemplate='<b>Split %{x}</b><br>HR: %{y:.0f} bpm<extra></extra>'
                ))
                
                fig_hr.update_layout(
                    title='Predicción de Frecuencia Cardíaca por Split',
                    xaxis_title='Split (km)',
                    yaxis_title='Frecuencia Cardíaca (bpm)',
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    font=dict(color='#333'),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                )
                
                st.plotly_chart(fig_hr, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"❤️ HR Máximo: {predictions_df['avg_hr'].max():.0f} bpm")
                with col2:
                    st.info(f"❤️ HR Mínimo: {predictions_df['avg_hr'].min():.0f} bpm")
            
            # Tab 4: Elevación
            with tab4:
                # Crear subplots
                fig_elev = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Ganancia de Elevación por Split", "Elevación Acumulada"),
                    vertical_spacing=0.12,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                )
                
                # Gráfico de barras de elevación
                fig_elev.add_trace(
                    go.Bar(
                        x=predictions_df['split_km'],
                        y=predictions_df['elevation_gain'],
                        name='Ganancia de Elevación',
                        marker=dict(color='#2ca02c'),
                        hovertemplate='<b>Split %{x}</b><br>Elevación: %{y:.0f}m<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Gráfico de elevación acumulada
                cumulative_elevation = predictions_df['elevation_gain'].cumsum()
                fig_elev.add_trace(
                    go.Scatter(
                        x=predictions_df['split_km'],
                        y=cumulative_elevation,
                        mode='lines+markers',
                        name='Elevación Acumulada',
                        line=dict(color='#9467bd', width=3),
                        marker=dict(size=8),
                        fill='tozeroy',
                        fillcolor='rgba(148, 103, 189, 0.3)',
                        hovertemplate='<b>Split %{x}</b><br>Acumulada: %{y:.0f}m<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig_elev.update_xaxes(title_text="Split (km)", row=2, col=1)
                fig_elev.update_yaxes(title_text="Elevación (m)", row=1, col=1)
                fig_elev.update_yaxes(title_text="Elevación Acumulada (m)", row=2, col=1)
                
                fig_elev.update_layout(
                    height=700,
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333'),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                    xaxis2=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                    yaxis2=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                    showlegend=True
                )
                
                st.plotly_chart(fig_elev, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📈 Total Elevación: {predictions_df['elevation_gain'].sum():.0f}m")
                with col2:
                    st.info(f"📉 Total Descenso: {predictions_df['elevation_loss'].sum():.0f}m")
            
            # Tab 5: Datos
            with tab5:
                st.subheader("📋 Tabla de Predicciones")
                
                # Mostrar tabla con opciones de visualización
                col1, col2 = st.columns([3, 1])
                with col2:
                    rows_to_show = st.selectbox(
                        "Filas a mostrar:",
                        [10, 25, 50, 100, len(predictions_df)],
                        index=0
                    )
                
                st.dataframe(
                    predictions_df.head(rows_to_show),
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("---")
                st.subheader("📥 Descargar Predicciones")
                
                # Botón de descarga
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar como CSV",
                    data=csv,
                    file_name=f"predicciones_{st.session_state.file_name.replace('.gpx', '.csv')}",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("---")
            st.subheader("🎯 Resumen de Predicción")
            
            # Calcular tiempo total
            tiempo_total = predictions_df['AVG_pace'].sum()
            horas = int(tiempo_total // 60)
            minutos = int(tiempo_total % 60)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"⏱️ **Tiempo Estimado**: {horas}h {minutos}m")
            with col2:
                st.info(f"📊 **Distancia Total**: ~{len(predictions_df)} km")
            with col3:
                dificultad = 'Alta' if predictions_df['elevation_gain'].sum() > 1000 else 'Media' if predictions_df['elevation_gain'].sum() > 500 else 'Baja'
                st.warning(f"🏔️ **Dificultad**: {dificultad}")
    
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("👆 Carga un archivo GPX para comenzar")
    
    # Mostrar información de bienvenida
    with st.container():
        st.markdown("""
        ## 🎯 ¿Cómo usar?
        
        1. **Carga tu GPX**: Arrastra o selecciona un archivo GPX de tu carrera
        2. **Genera Predicción**: Haz clic en el botón para procesar
        3. **Analiza Resultados**: Revisa gráficos interactivos, métricas y predicciones
        4. **Descarga**: Exporta los datos en formato CSV
        
        ## 📊 Información que obtendrás
        
        - ⏱️ **Ritmo por Split**: Predicción de ritmo para cada kilómetro
        - ❤️ **Frecuencia Cardíaca**: Estimación de HR durante la carrera
        - 🏔️ **Análisis de Elevación**: Ganancia y pérdida de altura
        - 📈 **Gráficos Interactivos**: Visualización completa del rendimiento con Plotly
        - ⏲️ **Tiempo Estimado**: Duración total de la carrera
        
        ## 🚀 Características
        
        - Análisis detallado por split de 1km
        - Visualización interactiva de tendencias
        - Cálculo automático de elevación acumulada
        - Exportación de datos en CSV
        - Interfaz intuitiva y responsiva
        - Gráficos interactivos con Plotly
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Smart Peak Splits © 2024 | Predictor de Rendimiento en Carreras</p>
    <p>Desarrollado con ❤️ usando Streamlit, TensorFlow, Plotly y análisis de datos</p>
</div>
""", unsafe_allow_html=True)
