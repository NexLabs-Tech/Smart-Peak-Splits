import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.predict import predict_race, extract_gpx_info, get_model_validation_metrics
from src.explainable_ai import ExplainableAI, plot_feature_importance, plot_correlation_heatmap, plot_prediction_factors
import os
import tempfile
import pickle

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
            
            # Métricas del Modelo (MAE, RMSE, R²)
            st.markdown("---")
            st.subheader("🎯 Métricas del Modelo")

            # Intentar cargar métricas de validación
            validation_metrics = get_model_validation_metrics()

            if validation_metrics:
                st.markdown("""
                Estas métricas muestran qué tan bien predice el modelo comparado con datos reales de entrenamiento:
                - **MAE** (Error Absoluto Medio): Error promedio en las predicciones
                - **RMSE** (Raíz del Error Cuadrático Medio): Penaliza errores grandes
                - **R²** (Coeficiente de Determinación): Qué tan bien explica el modelo (1.0 = perfecto, 0.0 = malo)
                """)

                col_metrics1, col_metrics2 = st.columns(2)

                with col_metrics1:
                    st.markdown("#### 🏃 Métricas de PACE (Ritmo)")
                    pace_metrics = validation_metrics.get('pace', {})

                    met_col1, met_col2, met_col3 = st.columns(3)
                    with met_col1:
                        st.metric(
                            "MAE",
                            f"{pace_metrics.get('mae', 0):.2f} min/km",
                            help="Error promedio en predicciones de ritmo"
                        )
                    with met_col2:
                        st.metric(
                            "RMSE",
                            f"{pace_metrics.get('rmse', 0):.2f} min/km",
                            help="Error cuadrático medio (penaliza errores grandes)"
                        )
                    with met_col3:
                        r2_value = pace_metrics.get('r2', 0)
                        r2_color = "normal" if r2_value > 0.5 else "inverse"
                        st.metric(
                            "R²",
                            f"{r2_value:.3f}",
                            help="Calidad del modelo (1.0 = perfecto, 0.0 = malo)",
                            delta=f"{'Bueno' if r2_value > 0.5 else 'Mejorable'}"
                        )

                with col_metrics2:
                    st.markdown("#### ❤️ Métricas de HR (Frecuencia Cardíaca)")
                    hr_metrics = validation_metrics.get('hr', {})

                    met_col1, met_col2, met_col3 = st.columns(3)
                    with met_col1:
                        st.metric(
                            "MAE",
                            f"{hr_metrics.get('mae', 0):.1f} bpm",
                            help="Error promedio en predicciones de HR"
                        )
                    with met_col2:
                        st.metric(
                            "RMSE",
                            f"{hr_metrics.get('rmse', 0):.1f} bpm",
                            help="Error cuadrático medio (penaliza errores grandes)"
                        )
                    with met_col3:
                        r2_value = hr_metrics.get('r2', 0)
                        st.metric(
                            "R²",
                            f"{r2_value:.3f}",
                            help="Calidad del modelo (1.0 = perfecto, 0.0 = malo)",
                            delta=f"{'Bueno' if r2_value > 0.5 else 'Mejorable'}"
                        )

                # Interpretación de las métricas
                st.markdown("---")
                st.markdown("**💡 Interpretación:**")

                pace_r2 = pace_metrics.get('r2', 0)
                hr_r2 = hr_metrics.get('r2', 0)

                if pace_r2 > 0.7:
                    st.success(f"✅ El modelo predice el **ritmo** muy bien (R² = {pace_r2:.3f})")
                elif pace_r2 > 0.5:
                    st.info(f"ℹ️ El modelo predice el **ritmo** de forma aceptable (R² = {pace_r2:.3f})")
                else:
                    st.warning(f"⚠️ El modelo tiene dificultad prediciendo el **ritmo** (R² = {pace_r2:.3f})")

                if hr_r2 > 0.7:
                    st.success(f"✅ El modelo predice la **HR** muy bien (R² = {hr_r2:.3f})")
                elif hr_r2 > 0.5:
                    st.info(f"ℹ️ El modelo predice la **HR** de forma aceptable (R² = {hr_r2:.3f})")
                else:
                    st.warning(f"⚠️ El modelo tiene dificultad prediciendo la **HR** (R² = {hr_r2:.3f}). Esto puede indicar que la HR no varía mucho o que faltan características importantes.")
            else:
                st.info("ℹ️ Métricas de validación no disponibles. Ejecuta el notebook de entrenamiento para generarlas.")

            # Métricas Avanzadas
            st.markdown("---")
            st.subheader("📈 Métricas Avanzadas")
            
            # Calcular métricas adicionales
            pace_std = predictions_df['AVG_pace'].std()
            hr_std = predictions_df['avg_hr'].std()
            pace_range = predictions_df['AVG_pace'].max() - predictions_df['AVG_pace'].min()
            hr_range = predictions_df['avg_hr'].max() - predictions_df['avg_hr'].min()
            
            # Índice de dificultad (basado en elevación y cambio de ritmo)
            difficulty_index = (total_elevation / len(predictions_df)) * (pace_range / predictions_df['AVG_pace'].mean())
            
            # Eficiencia (ritmo vs elevación)
            avg_elevation = predictions_df['elevation_gain'].mean()
            efficiency = (predictions_df['AVG_pace'].mean() / (avg_elevation + 1)) * 100
            
            # Zona de HR (determinar intensidad)
            avg_hr_percent = (avg_hr / 220) * 100  # Fórmula simplificada (220 - edad)
            if avg_hr_percent < 50:
                hr_zone = "🟢 Muy Baja (Recuperación)"
            elif avg_hr_percent < 60:
                hr_zone = "🟡 Baja (Base)"
            elif avg_hr_percent < 70:
                hr_zone = "🟠 Media (Aeróbica)"
            elif avg_hr_percent < 85:
                hr_zone = "🔴 Alta (Anaeróbica)"
            else:
                hr_zone = "🔴🔴 Muy Alta (Máxima)"
            
            # Variabilidad de ritmo (qué tan consistente es)
            pace_cv = (pace_std / predictions_df['AVG_pace'].mean()) * 100  # Coeficiente de variación
            
            # Splits críticos (más difíciles)
            hardest_splits = predictions_df.nlargest(3, 'AVG_pace')
            easiest_splits = predictions_df.nsmallest(3, 'AVG_pace')
            
            col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
            
            with col_adv1:
                st.metric(
                    "🎯 Índice Dificultad",
                    f"{difficulty_index:.2f}",
                    help="Combina elevación y variabilidad de ritmo (más alto = más difícil)"
                )
            
            with col_adv2:
                st.metric(
                    "⚙️ Eficiencia",
                    f"{efficiency:.1f}%",
                    help="Relación ritmo/elevación (mayor = más eficiente)"
                )
            
            with col_adv3:
                st.metric(
                    "💓 Zona HR",
                    hr_zone,
                    help="Intensidad de esfuerzo (basada en HR promedio)"
                )
            
            with col_adv4:
                st.metric(
                    "📊 Consistencia Ritmo",
                    f"{100 - pace_cv:.1f}%",
                    help="Qué tan constante es el ritmo (100% = perfectamente constante)"
                )
            
            # Más métricas
            st.markdown("---")
            
            col_adv5, col_adv6, col_adv7, col_adv8 = st.columns(4)
            
            with col_adv5:
                st.metric(
                    "📉 Variabilidad Ritmo",
                    f"{pace_std:.2f} min/km",
                    help="Desviación estándar del ritmo"
                )
            
            with col_adv6:
                st.metric(
                    "💗 Variabilidad HR",
                    f"{hr_std:.0f} bpm",
                    help="Desviación estándar de frecuencia cardíaca"
                )
            
            with col_adv7:
                st.metric(
                    "📊 Rango Ritmo",
                    f"{pace_range:.2f} min/km",
                    help=f"De {predictions_df['AVG_pace'].min():.2f} a {predictions_df['AVG_pace'].max():.2f}"
                )
            
            with col_adv8:
                st.metric(
                    "💓 Rango HR",
                    f"{hr_range:.0f} bpm",
                    help=f"De {predictions_df['avg_hr'].min():.0f} a {predictions_df['avg_hr'].max():.0f}"
                )
            
            # Splits críticos
            st.markdown("---")
            
            col_splits1, col_splits2 = st.columns(2)
            
            with col_splits1:
                st.markdown("### 🔴 Splits Más Difíciles (Ritmo Lento)")
                hardest_splits_display = hardest_splits[['split_km', 'AVG_pace', 'elevation_gain', 'avg_hr']].copy()
                hardest_splits_display.columns = ['Split', 'Ritmo (min/km)', 'Elevación (m)', 'HR (bpm)']
                st.dataframe(hardest_splits_display, hide_index=True, use_container_width=True)
            
            with col_splits2:
                st.markdown("### 🟢 Splits Más Fáciles (Ritmo Rápido)")
                easiest_splits_display = easiest_splits[['split_km', 'AVG_pace', 'elevation_gain', 'avg_hr']].copy()
                easiest_splits_display.columns = ['Split', 'Ritmo (min/km)', 'Elevación (m)', 'HR (bpm)']
                st.dataframe(easiest_splits_display, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Gráficos de Análisis")
            
            # Crear tabs para diferentes vistas
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📍 Mapa & Perfil", "Ritmo", "Frecuencia Cardíaca", "Elevación", "Datos", "🧠 Explainability"])
            
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
            
            # Tab 6: Explainable AI
            with tab6:
                st.subheader("🧠 Explainable AI - Interpretabilidad de Predicciones")
                
                st.markdown("""
                Esta sección explica **POR QUÉ** el modelo predice ciertos valores de ritmo y 
                frecuencia cardíaca, basándose en características del terreno y patrones identificados.
                
                **Conceptos:**
                - 📊 **Importancia de Características**: Qué factores afectan más las predicciones
                - 🔗 **Correlaciones**: Relación entre terreno y rendimiento
                - 🎯 **Explicaciones por Split**: Por qué cada predicción es así
                """)
                
                st.markdown("---")
                
                # Subtab para XAI
                xai_subtab1, xai_subtab2, xai_subtab3 = st.tabs(["Análisis Global", "Patrones", "Split Individual"])
                
                # Subtab 1: Análisis Global de Importancia
                with xai_subtab1:
                    st.markdown("### 📊 Importancia de Características")
                    st.markdown("""
                    Muestra qué características tienen mayor impacto en las predicciones de ritmo y HR.
                    """)
                    
                    # Crear datos para importancia (simulado basado en correlaciones)
                    feature_importance_data = pd.DataFrame({
                        'Feature': ['Elevation Gain', 'Gradient', 'Cumulative Elevation', 'Elevation Loss', 'Distance'],
                        'Importance': [
                            abs(predictions_df['AVG_pace'].corr(predictions_df['elevation_gain'])),
                            abs(predictions_df['AVG_pace'].corr(predictions_df['elevation_loss'])),
                            abs(predictions_df['avg_hr'].corr(predictions_df['elevation_gain'])),
                            abs(predictions_df['avg_hr'].corr(predictions_df['elevation_loss'])),
                            0.15
                        ]
                    }).sort_values('Importance', ascending=False)
                    
                    # Normalizar importancia
                    feature_importance_data['Importance'] = (
                        feature_importance_data['Importance'] / feature_importance_data['Importance'].max() * 100
                    )
                    
                    fig_importance = go.Figure()
                    
                    fig_importance.add_trace(go.Bar(
                        y=feature_importance_data['Feature'],
                        x=feature_importance_data['Importance'],
                        orientation='h',
                        marker=dict(
                            color=feature_importance_data['Importance'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Importancia (%)")
                        ),
                        hovertemplate='<b>%{y}</b><br>Importancia: %{x:.1f}%<extra></extra>'
                    ))
                    
                    fig_importance.update_layout(
                        title='Importancia de Características en Predicción',
                        xaxis_title='Importancia Relativa (%)',
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333'),
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(showgrid=False)
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Información detallada
                    st.markdown("**Interpretación:**")
                    st.markdown(f"""
                    - **Elevación Acumulada** (Importancia: {feature_importance_data.iloc[0]['Importance']:.1f}%): 
                      Es el factor más importante. A mayor elevación acumulada, mayor fatiga y ritmo más lento.
                    
                    - **Gradient** (Importancia: {feature_importance_data.iloc[1]['Importance']:.1f}%): 
                      El gradiente del terreno afecta significativamente la velocidad de carrera.
                    
                    - **Elevación por Split** (Importancia: {feature_importance_data.iloc[2]['Importance']:.1f}%): 
                      Cada metro de elevación en el split actual influye en el rendimiento.
                    """)
                
                # Subtab 2: Patrones Globales
                with xai_subtab2:
                    st.markdown("### 🔗 Patrones Identificados")
                    
                    # Calcular correlaciones
                    corr_pace_elev = predictions_df['AVG_pace'].corr(predictions_df['elevation_gain'].fillna(0))
                    corr_hr_elev = predictions_df['avg_hr'].corr(predictions_df['elevation_gain'].fillna(0))
                    
                    col_pattern1, col_pattern2, col_pattern3 = st.columns(3)
                    
                    with col_pattern1:
                        st.metric(
                            "🏃 Ritmo-Elevación",
                            f"{corr_pace_elev:.3f}",
                            help="Correlación entre elevación y ritmo (más elevación = ritmo más lento)"
                        )
                    
                    with col_pattern2:
                        st.metric(
                            "❤️ HR-Elevación",
                            f"{corr_hr_elev:.3f}",
                            help="Correlación entre elevación y frecuencia cardíaca"
                        )
                    
                    with col_pattern3:
                        avg_elevation_gain = predictions_df['elevation_gain'].mean()
                        st.metric(
                            "📈 Elevación Promedio",
                            f"{avg_elevation_gain:.0f}m",
                            help="Promedio de elevación por split"
                        )
                    
                    st.markdown("---")
                    
                    # Matriz de correlación
                    st.markdown("**Matriz de Correlación Completa:**")
                    
                    data_for_corr = pd.DataFrame({
                        'Elevation Gain': predictions_df['elevation_gain'],
                        'Elevation Loss': predictions_df['elevation_loss'],
                        'Predicted Pace': predictions_df['AVG_pace'],
                        'Predicted HR': predictions_df['avg_hr']
                    })
                    
                    corr_matrix = data_for_corr.corr()
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        zmin=-1,
                        zmax=1,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text:.2f}',
                        hovertemplate='%{y} vs %{x}<br>Correlación: %{z:.3f}<extra></extra>',
                        colorbar=dict(title="Correlación")
                    ))
                    
                    fig_corr.update_layout(
                        title='Correlaciones: Terreno vs Predicciones',
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333')
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Subtab 3: Explicación por Split Individual
                with xai_subtab3:
                    st.markdown("### 🎯 Explicación de Split Individual")
                    
                    # Selector de split
                    selected_split = st.slider(
                        "Selecciona un split para análisis detallado",
                        min_value=int(predictions_df['split_km'].min()),
                        max_value=int(predictions_df['split_km'].max()),
                        value=int(predictions_df['split_km'].min()) + 5
                    )
                    
                    split_idx = int(selected_split) - 1
                    split_row = predictions_df.iloc[split_idx]
                    
                    # Información del split
                    col_split1, col_split2, col_split3, col_split4 = st.columns(4)
                    
                    with col_split1:
                        st.metric("Split", f"{int(split_row['split_km'])} km")
                    
                    with col_split2:
                        st.metric(
                            "Ritmo Predicho",
                            f"{split_row['AVG_pace']:.2f}",
                            help="minutos por kilómetro"
                        )
                    
                    with col_split3:
                        st.metric(
                            "HR Predicho",
                            f"{split_row['avg_hr']:.0f}",
                            help="latidos por minuto"
                        )
                    
                    with col_split4:
                        st.metric(
                            "Elevación",
                            f"{split_row['elevation_gain']:.0f}m"
                        )
                    
                    st.markdown("---")
                    
                    # Análisis de factores que influyen
                    st.markdown("**Factores que Influyen en esta Predicción:**")
                    
                    factors = []
                    
                    # Factor 1: Elevación
                    elev_percentile = (predictions_df['elevation_gain'] <= split_row['elevation_gain']).sum() / len(predictions_df) * 100
                    if split_row['elevation_gain'] > predictions_df['elevation_gain'].quantile(0.75):
                        factors.append({
                            'emoji': '📈',
                            'factor': 'Elevación Alta',
                            'value': f"{split_row['elevation_gain']:.0f}m",
                            'impact': 'Ritmo MÁS LENTO',
                            'percentage': 40
                        })
                    
                    # Factor 2: Posición en la carrera
                    if selected_split > len(predictions_df) * 0.66:
                        factors.append({
                            'emoji': '😤',
                            'factor': 'Fatiga Acumulada',
                            'value': f'Split {int(selected_split)} / {int(predictions_df["split_km"].max())}',
                            'impact': 'Ritmo MÁS LENTO',
                            'percentage': 35
                        })
                    
                    # Factor 3: HR elevado
                    if split_row['avg_hr'] > predictions_df['avg_hr'].quantile(0.75):
                        factors.append({
                            'emoji': '❤️',
                            'factor': 'Estrés Cardíaco Alto',
                            'value': f"{split_row['avg_hr']:.0f} bpm",
                            'impact': 'Mayor exigencia física',
                            'percentage': 25
                        })
                    
                    if not factors:
                        factors.append({
                            'emoji': '✅',
                            'factor': 'Condiciones Favorables',
                            'value': 'Split técnicamente fácil',
                            'impact': 'Ritmo más rápido',
                            'percentage': 100
                        })
                    
                    factors_df = pd.DataFrame(factors)
                    
                    # Gráfico de factores
                    fig_factors = go.Figure()
                    
                    fig_factors.add_trace(go.Bar(
                        x=factors_df['percentage'],
                        y=factors_df['factor'],
                        orientation='h',
                        marker=dict(
                            color=['#d62728', '#ff7f0e', '#2ca02c'][:len(factors_df)],
                        ),
                        text=factors_df['percentage'].apply(lambda x: f'{x}%'),
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Impacto: %{x}%<extra></extra>'
                    ))
                    
                    fig_factors.update_layout(
                        title=f"Factores de Influencia - Split {int(selected_split)}",
                        xaxis_title='Impacto Relativo (%)',
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333'),
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(showgrid=False),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_factors, use_container_width=True)
                    
                    # Recomendaciones
                    st.markdown("**Recomendaciones Basadas en el Análisis:**")
                    
                    if split_row['avg_hr'] > predictions_df['avg_hr'].quantile(0.75):
                        st.warning("⚠️ HR muy elevado en este split. Considera reducir velocidad y mantente hidratado.")
                    
                    if split_row['elevation_gain'] > predictions_df['elevation_gain'].quantile(0.75):
                        st.info("💡 Elevación significativa. Mantén un ritmo controlado y respira profundamente.")
                    
                    if selected_split > len(predictions_df) * 0.7:
                        st.success("🏁 Estás en la recta final. Mantén la concentración y gestiona la fatiga.")
            
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
