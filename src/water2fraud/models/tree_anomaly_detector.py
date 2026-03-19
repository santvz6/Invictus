import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from src.config import DatasetKeys

class RegressionAnomalyDetector:
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, target_col: str, feature_cols: list, threshold_percentile: int = 99):
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"--- Entrenando Modelos para {target_col} ---")
        
        # 1. Inicialización y entrenamiento
        xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        
        xgb_reg.fit(X, y)
        rf_reg.fit(X, y)
        
        # 2. Predicciones y Residuos
        df_results = df.copy()
        df_results['prediccion_xgb'] = xgb_reg.predict(X)
        df_results['prediccion_rf'] = rf_reg.predict(X)
        
        df_results['error_xgb'] = np.abs(df_results[target_col] - df_results['prediccion_xgb'])
        df_results['error_rf'] = np.abs(df_results[target_col] - df_results['prediccion_rf'])
        
        # 3. Cálculo de Umbrales Independientes
        umbral_xgb = np.percentile(df_results['error_xgb'], threshold_percentile)
        umbral_rf = np.percentile(df_results['error_rf'], threshold_percentile)
        
        df_results['es_anomalia_xgb'] = df_results['error_xgb'] > umbral_xgb
        df_results['es_anomalia_rf'] = df_results['error_rf'] > umbral_rf
        
        # 4. Consenso (Opcional: ¿Están de acuerdo ambos modelos?)
        df_results['anomalia_consenso'] = df_results['es_anomalia_xgb'] & df_results['es_anomalia_rf']
        
        print(f"Anomalías XGBoost: {df_results['es_anomalia_xgb'].sum()}")
        print(f"Anomalías Random Forest: {df_results['es_anomalia_rf'].sum()}")
        print(f"Anomalías detectadas por AMBOS: {df_results['anomalia_consenso'].sum()}")
        
        return df_results, xgb_reg, rf_reg

    @staticmethod
    def plot_residuals(df_results, target_col, filtro_uso="TODOS"):
        plt.style.use('dark_background')
        
        # --- Lógica de filtrado ---
        col_uso = DatasetKeys.USO
        if filtro_uso != 'TODOS' and col_uso in df_results.columns:
            df_plot = df_results[df_results[col_uso] == filtro_uso].copy()
            titulo_extra = f"(Filtro: {filtro_uso})"
        else:
            df_plot = df_results.copy()
            titulo_extra = "(Todos los registros)"

        if df_plot.empty:
            print(f"No hay datos para el filtro: {filtro_uso}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        modelos = [
            ('xgb', ax1, 'XGBoost', '#3498db', '#e74c3c'),
            ('rf', ax2, 'Random Forest', '#2ecc71', '#f1c40f')
        ]
        
        for suffix, ax, name, c_norm, c_anom in modelos:
            # Segmentar normales y anomalías dentro del set filtrado
            normales = df_plot[~df_plot[f'es_anomalia_{suffix}']]
            anomalos = df_plot[df_plot[f'es_anomalia_{suffix}']]
            
            # Scatter
            ax.scatter(normales[f'prediccion_{suffix}'], normales[target_col], 
                       alpha=0.4, color=c_norm, label='Normal', s=20)
            ax.scatter(anomalos[f'prediccion_{suffix}'], anomalos[target_col], 
                       alpha=0.8, color=c_anom, label='Anomalía', s=40, edgecolors='white')
            
            # Línea ideal (bisectriz) basada en el dataset original para mantener perspectiva
            max_val = max(df_results[target_col].max(), df_results[f'prediccion_{suffix}'].max())
            ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', alpha=0.4)
            
            ax.set_title(f"{name} {titulo_extra}", fontsize=14)
            ax.set_xlabel(f"Predicción {name}")
            ax.set_ylabel(f"Real: {target_col}")
            ax.legend()
            ax.grid(True, alpha=0.1)

        plt.tight_layout()
        plt.show()