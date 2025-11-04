import numpy as np
import pandas as pd
import gpxpy
import pickle
import os
from tensorflow.keras.models import load_model


def extract_gpx_info(gpx_file_path):
    """Extrae información detallada del archivo GPX para visualización"""
    with open(gpx_file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    points_data = []
    total_distance = 0
    
    for track in gpx.tracks:
        for segment in track.segments:
            points = segment.points
            
            for i in range(len(points)):
                if i > 0:
                    prev_point = points[i-1]
                    curr_point = points[i]
                    distance = prev_point.distance_2d(curr_point)
                    total_distance += distance
                else:
                    distance = 0
                
                points_data.append({
                    'latitude': points[i].latitude,
                    'longitude': points[i].longitude,
                    'elevation': points[i].elevation if points[i].elevation else 0,
                    'cumulative_distance': total_distance / 1000,  # en km
                })
    
    df_points = pd.DataFrame(points_data)
    
    return {
        'points': df_points,
        'total_distance': total_distance / 1000,  # en km
        'min_elevation': df_points['elevation'].min(),
        'max_elevation': df_points['elevation'].max(),
        'avg_elevation': df_points['elevation'].mean(),
        'bounds': {
            'min_lat': df_points['latitude'].min(),
            'max_lat': df_points['latitude'].max(),
            'min_lon': df_points['longitude'].min(),
            'max_lon': df_points['longitude'].max(),
        }
    }


def parse_gpx(gpx_file_path):
    with open(gpx_file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    splits = []
    current_distance = 0
    split_km = 0
    
    for track in gpx.tracks:
        for segment in track.segments:
            points = segment.points
            
            split_elevation_gain = 0
            split_elevation_loss = 0
            prev_elevation = points[0].elevation if len(points) > 0 else 0
            
            for i in range(1, len(points)):
                prev_point = points[i-1]
                curr_point = points[i]
                
                distance = prev_point.distance_3d(curr_point)
                current_distance += distance
                
                elevation_diff = curr_point.elevation - prev_elevation
                if elevation_diff > 0:
                    split_elevation_gain += elevation_diff
                else:
                    split_elevation_loss += abs(elevation_diff)
                
                prev_elevation = curr_point.elevation
                
                if current_distance >= 1000:
                    splits.append({
                        'split_km': split_km + 1,
                        'distance': current_distance / 1000,
                        'elevation_gain': split_elevation_gain,
                        'elevation_loss': split_elevation_loss,
                    })
                    
                    current_distance = 0
                    split_elevation_gain = 0
                    split_elevation_loss = 0
                    split_km += 1
            
            if current_distance > 0:
                splits.append({
                    'split_km': split_km + 1,
                    'distance': current_distance / 1000,
                    'elevation_gain': split_elevation_gain,
                    'elevation_loss': split_elevation_loss,
                })
    
    df = pd.DataFrame(splits)
    df['cumulative_elevation_gain'] = df['elevation_gain'].cumsum()
    df['gradient'] = ((df['elevation_gain'] - df['elevation_loss']) / (df['distance'] * 1000 + 1e-6)) * 100
    
    return df


def predict_race(gpx_file_path, model_path=None, scaler_path=None):
    
    if model_path is None:
        possible_paths = [
            ('models/smart_peak_model.keras', 'models/scalers.pkl'),
            ('notebooks-research/models/best_model.keras', 'notebooks-research/models/scalers.pkl'),
            ('notebooks-research/models/smart_peak_model.keras', 'notebooks-research/models/scalers.pkl'),
        ]
        
        for mp, sp in possible_paths:
            if os.path.exists(mp) and os.path.exists(sp):
                model_path = mp
                scaler_path = sp
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"\n❌ Error: No se encontró el modelo entrenado\n"
                f"\nBuscado en:\n"
                f"  - models/smart_peak_model.keras\n"
                f"  - notebooks-research/models/best_model.keras\n"
                f"\nPrimero debes entrenar el modelo:\n"
                f"  1. Coloca tus CSVs en data/processed/\n"
                f"  2. Ejecuta: jupyter notebook train_model.ipynb\n"
                f"  3. Ejecuta todas las celdas del notebook\n"
                f"  4. Se generará models/smart_peak_model.keras\n"
            )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n❌ Error: No se encontró el modelo en '{model_path}'\n"
        )
    
    print(f"Procesando GPX: {gpx_file_path}")
    splits_df = parse_gpx(gpx_file_path)
    print(f"Splits extraídos: {len(splits_df)}")
    
    model = load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']
    sequence_length = scalers['sequence_length']
    feature_columns = scalers['feature_columns']
    
    X = splits_df[feature_columns].values
    X_scaled = scaler_X.transform(X)
    
    X_seq = []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
    X_seq = np.array(X_seq)
    
    if len(X_seq) == 0:
        print("Error: No hay suficientes splits para predecir")
        return None
    
    predictions_scaled = model.predict(X_seq, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    result_df = splits_df.copy()
    result_df['predicted_pace'] = np.nan
    result_df['predicted_hr'] = np.nan
    
    result_df.loc[sequence_length:, 'predicted_pace'] = predictions[:, 0]
    result_df.loc[sequence_length:, 'predicted_hr'] = predictions[:, 1]
    
    for i in range(sequence_length):
        result_df.loc[i, 'predicted_pace'] = predictions[0, 0]
        result_df.loc[i, 'predicted_hr'] = predictions[0, 1]
    
    output_df = result_df[['split_km', 'predicted_pace', 'elevation_gain', 'elevation_loss', 'predicted_hr']].copy()
    output_df.columns = ['split_km', 'AVG_pace', 'elevation_gain', 'elevation_loss', 'avg_hr']
    
    print(f"\nRESUMEN:")
    print(f"  Total splits: {len(output_df)}")
    print(f"  Pace promedio: {output_df['AVG_pace'].mean():.2f} min/km")
    print(f"  HR promedio: {output_df['avg_hr'].mean():.0f} bpm")
    print(f"  Elevación total: {output_df['elevation_gain'].sum():.0f}m")
    print(f"  Tiempo estimado: {output_df['AVG_pace'].sum():.2f} minutos")

    return output_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python predict.py <archivo.gpx> [output.csv]")
        sys.exit(1)
    
    gpx_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    
    predictions = predict_race(gpx_path)
    
    if predictions is not None:
        predictions.to_csv(output_path, index=False)
        print(f"\nPredicciones guardadas en: {output_path}")
        
        print("\nPrimeros 10 splits:")
        print(predictions.head(10))