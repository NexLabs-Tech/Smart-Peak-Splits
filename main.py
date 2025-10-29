import sys
import os
from src.predict import predict_race

def main():
    print("="*60)
    print("SMART PEAK SPLITS - Race Prediction")
    print("="*60)
    print()
    
    if len(sys.argv) < 2:
        print("Uso: python main.py <archivo.gpx> [output.csv]")
        print()
        print("Ejemplos:")
        print("  python main.py mi_carrera.gpx")
        print("  python main.py mi_carrera.gpx resultados.csv")
        print()
        sys.exit(1)
    
    gpx_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    
    if not os.path.exists(gpx_file):
        print(f"Error: El archivo '{gpx_file}' no existe")
        sys.exit(1)
    
    if not os.path.exists('./notebooks-research/models/smart_peak_model.keras'):
        print("Error: No se encontró el modelo entrenado")
        print("Por favor ejecuta primero 'train_model.ipynb'")
        sys.exit(1)
    
    print(f"Archivo GPX: {gpx_file}")
    print(f"Output: {output_file}")
    print()
    
    predictions = predict_race(gpx_file)
    
    if predictions is not None:
        predictions.to_csv(output_file, index=False)
        print(f"\n✓ Predicciones guardadas en: {output_file}")
        
        print("\n" + "="*60)
        print("PREDICCIONES (primeros 10 splits)")
        print("="*60)
        print(predictions.head(10).to_string(index=False))
        print()
        print(f"Total de {len(predictions)} splits predichos")
    else:
        print("\nError: No se pudo generar predicciones")
        sys.exit(1)

if __name__ == "__main__":
    main()