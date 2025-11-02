import os
import sys
import argparse
import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from PIL import Image
import io

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict_race, parse_gpx


def get_gpx_track_points(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation,
                    'time': point.time
                })
    return pd.DataFrame(points)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radio terrestre en metros
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def prepare_animation_data(gpx_file, predictions_file):
    # Cargar puntos del GPX
    track_points = get_gpx_track_points(gpx_file)

    # Cargar predicciones
    predictions = pd.read_csv(predictions_file)

    # Calcular distancia acumulada
    distances = [0]
    for i in range(1, len(track_points)):
        d = haversine_distance(
            track_points.iloc[i-1]['lat'], track_points.iloc[i-1]['lon'],
            track_points.iloc[i]['lat'], track_points.iloc[i]['lon']
        )
        distances.append(distances[-1] + d)

    track_points['distance'] = distances

    return track_points, predictions


def create_animation(gpx_file, predictions_file, output_gif='animation.gif', fps=10, frames=None):
    print(f"Preparando datos de animación...")
    track_points, predictions = prepare_animation_data(gpx_file, predictions_file)

    if frames:
        track_points = track_points.iloc[:frames]

    total_distance = track_points['distance'].max()
    n_frames = len(track_points)

    print(f"Total de frames: {n_frames}")
    print(f"Distancia total: {total_distance/1000:.2f} km")
    print(f"Generando animación...")

    fig = plt.figure(figsize=(16, 10), dpi=100)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Mapa de la ruta
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_map.set_title('Ruta de la Carrera', fontsize=14, fontweight='bold')
    ax_map.plot(track_points['lon'], track_points['lat'], 'b-', alpha=0.5, linewidth=1)
    ax_map.scatter(track_points['lon'].iloc[0], track_points['lat'].iloc[0],
                   color='green', s=100, marker='o', label='Inicio', zorder=5)
    ax_map.scatter(track_points['lon'].iloc[-1], track_points['lat'].iloc[-1],
                   color='red', s=100, marker='s', label='Meta', zorder=5)
    ax_map.set_xlabel('Longitud')
    ax_map.set_ylabel('Latitud')
    ax_map.legend()
    ax_map.grid(True, alpha=0.3)

    # Placeholder para marcador de posición actual
    marker_map, = ax_map.plot([], [], 'r*', markersize=20, label='Posición actual')

    # Subplot 2: Elevación
    ax_ele = fig.add_subplot(gs[0, 1])
    ax_ele.set_title('Elevación', fontsize=12, fontweight='bold')
    ax_ele.plot(track_points['distance']/1000, track_points['ele'], 'gray', alpha=0.5, linewidth=1)
    ax_ele.fill_between(track_points['distance']/1000, track_points['ele'], alpha=0.3, color='brown')
    ax_ele.set_xlabel('Distancia (km)')
    ax_ele.set_ylabel('Elevación (m)')
    ax_ele.grid(True, alpha=0.3)
    line_ele, = ax_ele.plot([], [], 'r-', linewidth=2)

    # Subplot 3: Pace predicho
    ax_pace = fig.add_subplot(gs[1, 1])
    ax_pace.set_title('Pace Predicho', fontsize=12, fontweight='bold')
    ax_pace.set_xlabel('Split (km)')
    ax_pace.set_ylabel('Pace (min/km)')
    if len(predictions) > 0 and 'AVG_pace' in predictions.columns:
        ax_pace.plot(predictions.index, predictions['AVG_pace'], 'b-o', linewidth=2, markersize=4)
        ax_pace.set_ylim(min(predictions['AVG_pace']) * 0.95, max(predictions['AVG_pace']) * 1.05)
    ax_pace.grid(True, alpha=0.3)
    line_pace, = ax_pace.plot([], [], 'r*', markersize=15)

    # Subplot 4: HR predicho
    ax_hr = fig.add_subplot(gs[2, 1])
    ax_hr.set_title('Frecuencia Cardíaca Predicha', fontsize=12, fontweight='bold')
    ax_hr.set_xlabel('Split (km)')
    ax_hr.set_ylabel('HR (bpm)')
    if len(predictions) > 0 and 'avg_hr' in predictions.columns:
        ax_hr.plot(predictions.index, predictions['avg_hr'], 'r-o', linewidth=2, markersize=4)
        ax_hr.set_ylim(min(predictions['avg_hr']) * 0.95, max(predictions['avg_hr']) * 1.05)
    ax_hr.grid(True, alpha=0.3)
    line_hr, = ax_hr.plot([], [], 'g*', markersize=15)

    # Subplot 5: Información en tiempo real
    ax_info = fig.add_subplot(gs[2, 0])
    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.5, '', fontsize=11, verticalalignment='center',
                            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    frames_data = []
    for i in range(n_frames):
        # Calcular split actual
        current_distance_km = track_points.iloc[i]['distance'] / 1000
        current_split = int(current_distance_km)

        # Obtener predicción si existe
        pace_pred = None
        hr_pred = None
        if current_split < len(predictions):
            pace_pred = predictions.iloc[current_split]['AVG_pace'] if 'AVG_pace' in predictions.columns else None
            hr_pred = predictions.iloc[current_split]['avg_hr'] if 'avg_hr' in predictions.columns else None

        frames_data.append({
            'lat': track_points.iloc[i]['lat'],
            'lon': track_points.iloc[i]['lon'],
            'ele': track_points.iloc[i]['ele'],
            'distance_km': current_distance_km,
            'split': current_split,
            'pace_pred': pace_pred,
            'hr_pred': hr_pred
        })

    def animate(frame_idx):
        if frame_idx >= len(frames_data):
            return

        data = frames_data[frame_idx]

        # Actualizar marcador en el mapa
        marker_map.set_data([data['lon']], [data['lat']])

        # Actualizar línea de elevación
        mask = track_points['distance'] <= data['distance_km'] * 1000
        if mask.sum() > 0:
            line_ele.set_data(track_points.loc[mask, 'distance'] / 1000,
                            track_points.loc[mask, 'ele'])

        # Actualizar marcadores de predicción
        if data['pace_pred'] is not None:
            line_pace.set_data([data['split']], [data['pace_pred']])

        if data['hr_pred'] is not None:
            line_hr.set_data([data['split']], [data['hr_pred']])

        # Actualizar información
        info = f"""
╔═══════════════════════════════════╗
║     INFORMACIÓN EN TIEMPO REAL     ║
╠═══════════════════════════════════╣
║ Distancia: {data['distance_km']:>22.2f} km ║
║ Elevación: {data['ele']:>22.0f} m  ║
║ Split: {data['split']:>28} ║
║                                   ║
║ Pace Predicho: {str(data['pace_pred'])[:12]:>19} ║
║ HR Predicho: {str(data['hr_pred'])[:12]:>23} ║
╚═══════════════════════════════════╝
        """
        info_text.set_text(info)

        return marker_map, line_ele, line_pace, line_hr, info_text

    # Crear animación
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data),
        interval=100, blit=True, repeat=True
    )

    # Guardar como GIF
    print(f"Guardando animación a {output_gif}...")
    anim.save(output_gif, writer='pillow', fps=fps, dpi=80)
    print(f"Animación guardada: {output_gif}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Crea una animación GIF de la carrera basada en predicciones LSTM'
    )
    parser.add_argument('gpx_file', help='Archivo GPX de entrada')
    parser.add_argument('-o', '--output', default='animation.gif', help='Archivo GIF de salida (default: animation.gif)')
    parser.add_argument('-p', '--predictions', help='Archivo CSV de predicciones (si no se proporciona, se genera)')
    parser.add_argument('--fps', type=int, default=10, help='FPS de la animación (default: 10)')
    parser.add_argument('--frames', type=int, help='Limitar número de frames (opcional)')

    args = parser.parse_args()

    if not os.path.exists(args.gpx_file):
        print(f"Error: No se encontró {args.gpx_file}")
        sys.exit(1)

    if not args.predictions:
        print(f"Generando predicciones para {args.gpx_file}...")
        args.predictions = 'temp_predictions.csv'
        try:
            predict_race(args.gpx_file, args.predictions)
        except Exception as e:
            print(f"Error al generar predicciones: {e}")
            sys.exit(1)

    if not os.path.exists(args.predictions):
        print(f"Error: No se encontró {args.predictions}")
        sys.exit(1)

    # Crear animación
    create_animation(args.gpx_file, args.predictions, args.output, args.fps, args.frames)


if __name__ == '__main__':
    main()
