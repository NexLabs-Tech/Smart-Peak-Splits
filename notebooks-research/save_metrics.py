"""
Script para calcular y guardar métricas de validación del modelo.
Ejecutar este script después de entrenar el modelo en el notebook.
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath('..'))
from src.predict import calculate_model_metrics

def save_validation_metrics(model_path, X_val, y_val, scaler_y, output_path=None):
    """
    Calcula y guarda las métricas de validación del modelo.

    Args:
        model_path: Path al modelo entrenado (.keras)
        X_val: Datos de validación de entrada
        y_val: Datos de validación de salida (targets)
        scaler_y: Scaler usado para normalizar los targets
        output_path: Path donde guardar las métricas (por defecto: mismo directorio que el modelo)
    """
    print("Cargando modelo...")
    model = load_model(model_path)

    print("Generando predicciones en conjunto de validación...")
    predictions_scaled = model.predict(X_val, verbose=0)

    print("Desnormalizando predicciones...")
    y_val_unscaled = scaler_y.inverse_transform(y_val)
    predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)

    print("Calculando métricas...")
    metrics = calculate_model_metrics(y_val_unscaled, predictions_unscaled)

    # Guardar métricas
    if output_path is None:
        output_path = model_path.replace('.keras', '_metrics.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\n✅ Métricas guardadas en: {output_path}\n")
    print("=" * 60)
    print("MÉTRICAS DE VALIDACIÓN")
    print("=" * 60)
    print("\n🏃 PACE (Ritmo):")
    print(f"  MAE:  {metrics['pace']['mae']:.2f} min/km")
    print(f"  RMSE: {metrics['pace']['rmse']:.2f} min/km")
    print(f"  R²:   {metrics['pace']['r2']:.3f}")

    print("\n❤️  HR (Frecuencia Cardíaca):")
    print(f"  MAE:  {metrics['hr']['mae']:.1f} bpm")
    print(f"  RMSE: {metrics['hr']['rmse']:.1f} bpm")
    print(f"  R²:   {metrics['hr']['r2']:.3f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    print("""
    Este script debe ejecutarse después de entrenar el modelo.

    Para usarlo desde el notebook, añade al final:

    ```python
    from save_metrics import save_validation_metrics

    metrics = save_validation_metrics(
        model_path='models/smart_peak_model.keras',
        X_val=X_val,
        y_val=y_val,
        scaler_y=scaler_y
    )
    ```
    """)
