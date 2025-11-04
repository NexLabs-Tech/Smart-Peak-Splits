"""
Módulo de Explainable AI para Smart Peak Splits
Basado en interpretabilidad y análisis de características
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ExplainableAI:
    """
    Clase para proporcionar explicaciones interpretables de las predicciones del modelo
    Basado en conceptos de Explainability, Interpretability y Transparency
    """
    
    def __init__(self, model, scaler_X, scaler_y, feature_columns):
        """
        Inicializa el módulo de XAI
        
        Args:
            model: Modelo TensorFlow entrenado
            scaler_X: Escalador de características
            scaler_y: Escalador de variables objetivo
            feature_columns: Lista de nombres de características
        """
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_columns = feature_columns
        self.feature_importance = None
        
    def calculate_feature_importance(self, X, y):
        """
        Calcula la importancia de cada característica usando permutación
        
        Args:
            X: Características (array numpy)
            y: Variables objetivo (array numpy)
            
        Returns:
            DataFrame con importancia de características
        """
        # Calcular importancia por permutación
        result = permutation_importance(
            self.model, X, y, 
            n_repeats=10, 
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df
    
    def get_split_explanation(self, split_data, prediction, split_km):
        """
        Genera explicación interpretable para una predicción específica de un split
        
        Args:
            split_data: Datos del split actual
            prediction: Predicción (pace, hr)
            split_km: Número del split
            
        Returns:
            Dict con explicación detallada
        """
        explanation = {
            'split_km': split_km,
            'predicted_pace': prediction[0],
            'predicted_hr': prediction[1],
            'factors': []
        }
        
        # Analizar factores que influyen en la predicción
        if split_data['elevation_gain'] > split_data['elevation_gain'].quantile(0.75):
            explanation['factors'].append({
                'factor': '📈 Elevación Alta',
                'value': f"{split_data['elevation_gain']:.0f}m",
                'impact': 'Aumenta ritmo (más lento)',
                'percentage': 35
            })
        
        if split_data['gradient'] > 5:
            explanation['factors'].append({
                'factor': '🏔️ Gradiente Técnico',
                'value': f"{split_data['gradient']:.1f}%",
                'impact': 'Aumenta fatiga',
                'percentage': 25
            })
        
        if split_km > 5 and split_data['cumulative_elevation_gain'] > 500:
            explanation['factors'].append({
                'factor': '😤 Fatiga Acumulada',
                'value': f"{split_data['cumulative_elevation_gain']:.0f}m acumulados",
                'impact': 'Ritmo más lento',
                'percentage': 30
            })
        
        if split_data['elevation_loss'] > 100:
            explanation['factors'].append({
                'factor': '📉 Descenso Fuerte',
                'value': f"{split_data['elevation_loss']:.0f}m",
                'impact': 'Mayor estrés cardíaco',
                'percentage': 10
            })
        
        return explanation
    
    def get_global_patterns(self, splits_df, predictions_df):
        """
        Identifica patrones globales en las predicciones
        
        Args:
            splits_df: DataFrame con información de splits
            predictions_df: DataFrame con predicciones
            
        Returns:
            Dict con patrones identificados
        """
        patterns = {
            'pace_elevation_correlation': splits_df['elevation_gain'].corr(predictions_df['AVG_pace']),
            'hr_elevation_correlation': splits_df['elevation_gain'].corr(predictions_df['avg_hr']),
            'hardest_splits': [],
            'easiest_splits': [],
            'risk_zones': []
        }
        
        # Identificar splits más difíciles (mayor elevación + mayor ritmo)
        hardest = predictions_df.nlargest(3, 'AVG_pace')
        for idx, row in hardest.iterrows():
            patterns['hardest_splits'].append({
                'split': int(row['split_km']),
                'pace': row['AVG_pace'],
                'elevation': splits_df.iloc[idx]['elevation_gain']
            })
        
        # Identificar splits más fáciles (menor ritmo)
        easiest = predictions_df.nsmallest(3, 'AVG_pace')
        for idx, row in easiest.iterrows():
            patterns['easiest_splits'].append({
                'split': int(row['split_km']),
                'pace': row['AVG_pace'],
                'elevation': splits_df.iloc[idx]['elevation_gain']
            })
        
        # Identificar zonas de riesgo (HR muy alto + elevación)
        high_hr_threshold = predictions_df['avg_hr'].quantile(0.75)
        high_elev_threshold = splits_df['elevation_gain'].quantile(0.75)
        
        risk_indices = (
            (predictions_df['avg_hr'] > high_hr_threshold) & 
            (splits_df['elevation_gain'] > high_elev_threshold)
        ).values
        
        for idx in np.where(risk_indices)[0]:
            patterns['risk_zones'].append({
                'split': int(predictions_df.iloc[idx]['split_km']),
                'hr': predictions_df.iloc[idx]['avg_hr'],
                'elevation': splits_df.iloc[idx]['elevation_gain'],
                'recommendation': '⚠️ Zona de riesgo: mantener hidratación y ritmo controlado'
            })
        
        return patterns
    
    def get_model_confidence(self, predictions_scaled, predictions):
        """
        Calcula un score de confianza en las predicciones
        
        Args:
            predictions_scaled: Predicciones sin escalar
            predictions: Predicciones escaladas
            
        Returns:
            Dict con información de confianza
        """
        # Usar la varianza de las predicciones como proxy de confianza
        pace_var = np.var(predictions[:, 0])
        hr_var = np.var(predictions[:, 1])
        
        confidence = {
            'pace_stability': max(0, 100 - pace_var * 10),  # Score 0-100
            'hr_stability': max(0, 100 - hr_var * 10),
            'overall_confidence': (max(0, 100 - pace_var * 10) + max(0, 100 - hr_var * 10)) / 2
        }
        
        return confidence


def plot_feature_importance(importance_df):
    """
    Crea gráfico de importancia de características
    
    Args:
        importance_df: DataFrame con importancia de características
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h',
        name='Importancia',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importancia")
        ),
        error_x=dict(
            type='data',
            array=importance_df['Std'],
            visible=True
        ),
        hovertemplate='<b>%{y}</b><br>Importancia: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Importancia de Características en Predicción',
        xaxis_title='Importancia (Permutación)',
        yaxis_title='Características',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
        yaxis=dict(showgrid=False)
    )
    
    return fig


def plot_correlation_heatmap(splits_df, predictions_df):
    """
    Crea mapa de correlaciones entre características y predicciones
    
    Args:
        splits_df: DataFrame con características
        predictions_df: DataFrame con predicciones
        
    Returns:
        Figura Plotly
    """
    # Preparar datos
    data_to_correlate = pd.DataFrame({
        'Elevation Gain': splits_df['elevation_gain'],
        'Elevation Loss': splits_df['elevation_loss'],
        'Gradient': splits_df['gradient'],
        'Predicted Pace': predictions_df['AVG_pace'],
        'Predicted HR': predictions_df['avg_hr']
    })
    
    corr_matrix = data_to_correlate.corr()
    
    fig = go.Figure(data=go.Heatmap(
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
    
    fig.update_layout(
        title='Matriz de Correlación: Características vs Predicciones',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333')
    )
    
    return fig


def plot_prediction_factors(explanation):
    """
    Visualiza factores que influyen en una predicción específica
    
    Args:
        explanation: Dict con explicación del split
        
    Returns:
        Figura Plotly
    """
    if not explanation['factors']:
        return None
    
    factors = pd.DataFrame(explanation['factors'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=factors['percentage'],
        y=factors['factor'],
        orientation='h',
        marker=dict(
            color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'],
            line=dict(width=0)
        ),
        text=factors['percentage'].apply(lambda x: f'{x}%'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Impacto: %{x}%<br>%{customdata}<extra></extra>',
        customdata=factors['impact']
    ))
    
    fig.update_layout(
        title=f"Factores de Influencia - Split {explanation['split_km']}",
        xaxis_title='Impacto Relativo (%)',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
        yaxis=dict(showgrid=False),
        showlegend=False
    )
    
    return fig
