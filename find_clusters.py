import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from src.config import AIConstants, Paths
from src.water2fraud.features.preprocessor import WaterPreprocessor

def find_optimal_clusters(df_raw, max_clusters=10):
    print("Preprocesando datos para el análisis del codo...")
    
    # 1. Preprocesar igual que en tu pipeline
    df_clean = WaterPreprocessor.process_raw_data(df_raw)
    X_sequences, _ = WaterPreprocessor.create_sequences(df_clean, sequence_length=12)
    
    # 2. Extraer solo la feature que usamos para agrupar (ej. CONSUMO_RATIO, índice 0)
    X_target = X_sequences[:, :, 0].reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
    
    # 3. Escalar las series
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X_target)
    
    # 4. Calcular inercia para cada K
    inertias = []
    K = range(1, max_clusters + 1)
    
    print(f"Calculando K-Means con DTW desde k=1 hasta k={max_clusters} (Esto puede tardar un poco)...")
    for k in K:
        km = TimeSeriesKMeans(
            n_clusters=k, 
            metric="dtw", 
            n_jobs=-1, 
            random_state=AIConstants.RANDOM_STATE
        )
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"  k={k} -> Inercia: {km.inertia_:.2f}")
        
    # 5. Dibujar la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clústeres (k)', fontsize=12)
    plt.ylabel('Inercia (Distancia DTW Acumulada)', fontsize=12)
    plt.title('Método del Codo para Series Temporales de Agua', fontsize=14)
    plt.xticks(K)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar la gráfica en tu carpeta de experimentos
    output_path = Paths.EXPERIMENTS_DIR / "metodo_del_codo.png"
    plt.savefig(output_path)
    print(f"\n¡Gráfica guardada en {output_path}!")
    print("Abre la imagen y busca el 'codo' (el punto donde la curva deja de caer en picado y se aplana).")

if __name__ == "__main__":
    import pandas as pd
    # Cargar tus datos y ejecutar
    df = pd.read_csv(Paths.RAW_CSV_AMAEM)
    find_optimal_clusters(df)