"""
Módulo de cálculo de consumo físico esperado mediante modelado híbrido.

Este componente combina un motor matemático basado en series de Fourier para 
capturar la estacionalidad física natural del agua, con un modelo de Machine 
Learning (Random Forest) que analiza el impacto de factores externos (clima, 
turismo, etc.) sobre el residuo estacional.
"""

import numpy as np
import pandas as pd
import logging
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor

from src.config import DatasetKeys, Paths, get_logger, AIConstants, FeatureConfig

# Configuración del logger para seguimiento del modelado físico
logger = get_logger(__name__)

class ModeloFisico:
    """
    Procesador de modelado híbrido para la determinación del consumo base.
    
    Utiliza una aproximación de dos etapas:
    1. Base Física: Onda estacional de Fourier ajustada por Barrio y Uso.
    2. Componente de Impacto: Modelado de residuos mediante factores exógenos.
    """

    NIVEL_ALERTAS = [
        '1_EXCESO_Grave', '2_EXCESO_Moderado', '3_EXCESO_Leve',
        '4_DEFECTO_Grave', '5_DEFECTO_Moderado', '6_DEFECTO_Leve'
    ]

    @staticmethod
    def _modelo_fourier(t, m, c, a1, b1, a2, b2):
        """
        Ecuación de Fourier de segundo orden con componente tendencial.
        Define la 'huella dactilar' del consumo físico esperado bajo condiciones ideales.
        """
        w = 2 * np.pi / 12  # Frecuencia fundamental (anual)
        return (m * t + c) + (a1 * np.cos(w * t) + b1 * np.sin(w * t)) + (a2 * np.cos(2 * w * t) + b2 * np.sin(2 * w * t))

    @staticmethod
    def process(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, RandomForestRegressor, list[str]]:
        """
        Ejecuta el pipeline de cálculo del consumo físico esperado.

        Args:
            df (pd.DataFrame): Dataset enriquecido con variables externas.
            feature_names (list[str]): Lista de columnas exógenas a considerar.

        Returns:
            pd.DataFrame: Dataset con columnas de predicción física y residuos de anomalía:
                - PREDICCION_FOURIER
                - IMPACTO_EXOGENO
                - RESIDUO
                - CONSUMO_FISICO_ESPERADO
        """
        logger.info("Iniciando cálculo de consumo físico esperado (Fourier + ML)...")
        
        # 1. Preparación y ordenación cronológica
        df = ModeloFisico._prepare_data(df)

        # 2. Fase de Fourier: Estacionalidad base por segmento [Barrio x Uso]
        df = ModeloFisico._calculate_fourier_baseline(df)

        # 3. Fase de ML: Modelado del impacto de variables exógenas
        df, rf_model, features_rf = ModeloFisico._calculate_ml_impact(df, feature_names)

        # 4. Consolidación y persistencia
        df = ModeloFisico._finalize_fisicos(df)

        return df, rf_model, features_rf

    @staticmethod
    def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """Garantiza la integridad temporal necesaria para el ajuste de curvas."""
        df_copy = df.copy()
        df_copy[DatasetKeys.FECHA] = pd.to_datetime(df_copy[DatasetKeys.FECHA])
        return df_copy.sort_values(by=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA])

    @staticmethod
    def _calculate_fourier_baseline(df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta una onda estacional independiente para cada combinación de Barrio y Uso."""
        df[DatasetKeys.PREDICCION_FOURIER] = 0.0

        for (barrio, uso), group in df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO]):
            mask = (df[DatasetKeys.BARRIO] == barrio) & (df[DatasetKeys.USO] == uso)
            
            # Evitamos Data Leakage: Ajustamos Fourier SOLO con datos de 2022-2023
            train_mask = group[DatasetKeys.FECHA].dt.year <= 2023
            
            y_target_all = group[DatasetKeys.CONSUMO_RATIO].values
            t_arr_all = np.arange(len(y_target_all))
            
            y_target_train = y_target_all[train_mask]
            t_arr_train = t_arr_all[train_mask]
            
            try:
                if len(y_target_train) > 0:
                    coef, _ = curve_fit(
                        ModeloFisico._modelo_fourier, t_arr_train, y_target_train, 
                        p0=[0, np.mean(y_target_train), 1000, 1000, 100, 100], maxfev=10000
                    )
                else:
                    raise ValueError("Sin datos de entrenamiento")
                # Predecimos para todos los meses (incluido 2024)
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = ModeloFisico._modelo_fourier(t_arr_all, *coef)
            except Exception:
                # Fallback: Si el ajuste falla por falta de datos, se usa el valor medio histórico
                logger.warning(f"Fallo en ajuste Fourier para {barrio} - {uso}. Aplicando media.")
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = np.mean(y_target_train) if len(y_target_train) > 0 else np.mean(y_target_all)
        
        return df

    @staticmethod
    def _calculate_ml_impact(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, RandomForestRegressor, list[str]]:
        """Entrena un modelo para predecir cuánto del consumo depende de factores externos."""
        # Cálculo del residuo estacional (lo que Fourier no explica)
        df[DatasetKeys.RESIDUO] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.PREDICCION_FOURIER]

        # Limpieza de variables exógenas
        exogenas = [col for col in feature_names if col in df.columns and col != DatasetKeys.CONSUMO_RATIO]
        for col in exogenas:
            df[col] = df[col].fillna(df[col].mean())

        # Generamos contextos de uso (DOMESTICO vs OTROS) - es de baja cardinalidad, seguro contra leakage masivo
        df_ml = pd.get_dummies(df, columns=[DatasetKeys.USO])
        columnas_contexto = [col for col in df_ml.columns if col.startswith(DatasetKeys.USO + '_')]
        
        features_rf = exogenas + [DatasetKeys.PREDICCION_FOURIER] + columnas_contexto
        X = df_ml[features_rf].fillna(0)
        y = df_ml[DatasetKeys.RESIDUO].fillna(0)
        
        # Evitamos Data Leakage: Random Forest SOLO ve el contexto histórico (2022-2023)
        train_mask_ml = pd.to_datetime(df[DatasetKeys.FECHA]).dt.year <= 2023
        X_train, y_train = X[train_mask_ml], y[train_mask_ml]
        
        if len(X_train) == 0:
            logger.warning("No hay datos de 2022-2023 para el RF Físico. Entrenando con todo.")
            X_train, y_train = X, y
            
        ml_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=AIConstants.RANDOM_STATE, n_jobs=-1)
        ml_model.fit(X_train, y_train)
        
        df[DatasetKeys.IMPACTO_EXOGENO] = ml_model.predict(X) # Predice todo, incluido 2024
        return df, ml_model, features_rf

    @staticmethod
    def _finalize_fisicos(df: pd.DataFrame) -> pd.DataFrame:
        """Combina componentes y genera los residuos finales para la detección de fraude y porcentajes de causa."""
        # Híbrido: Base Física + Impacto de Contexto
        df[DatasetKeys.CONSUMO_FISICO_ESPERADO] = df[DatasetKeys.PREDICCION_FOURIER] + df[DatasetKeys.IMPACTO_EXOGENO]
        
        # Residuo Final: Diferencia entre consumo real y lo que la 'física' y el 'contexto' dictan
        # Un residuo positivo elevado es un fuerte indicador de posible fraude/fuga
        df[DatasetKeys.RESIDUO] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.CONSUMO_FISICO_ESPERADO]
        
        # --- 4. Cálculo de Causas (Porcentajes) y Triaje de Anomalías ---
        sospechosos_posibles = FeatureConfig.CAUSAS_EXOGENAS
        
        sospechosos = {k: v for k, v in sospechosos_posibles.items() if k in df.columns}
        
        # Función auxiliar para cálculo robusto de Z-Score evitando avisos de división por cero
        def robust_zscore(x):
            std = x.std()
            if std == 0 or np.isnan(std):
                return 0.0
            return (x - x.mean()) / std

        for var in sospechosos.keys():
            df[f'z_{var}'] = df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO])[var].transform(robust_zscore)
            df[f'peso_{var}'] = df[f'z_{var}'].abs()
            
        df[DatasetKeys.Z_ERROR_FINAL] = df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO])[DatasetKeys.RESIDUO].transform(robust_zscore)
        df['peso_Desconocido'] = df[DatasetKeys.Z_ERROR_FINAL].abs()
        
        columnas_pesos = [f'peso_{var}' for var in sospechosos.keys()] + ['peso_Desconocido']
        df['suma_pesos'] = df[columnas_pesos].sum(axis=1).replace(0, 1) # Evitar división por cero
        
        columnas_pct = []
        for var, col_name in sospechosos.items():
            df[col_name] = (df[f'peso_{var}'] / df['suma_pesos']) * 100
            columnas_pct.append(col_name)
            
        df[DatasetKeys.PCT_CAUSA_DESCONOCIDA] = (df['peso_Desconocido'] / df['suma_pesos']) * 100
        columnas_pct.append(DatasetKeys.PCT_CAUSA_DESCONOCIDA)
        
        # --- Umbrales del Semáforo (Niveles de Alerta) ---
        z_leve = 1.5
        z_mod = 2.0
        z_grave = 2.5
        
        condiciones = [
            df[DatasetKeys.Z_ERROR_FINAL] > z_grave,
            (df[DatasetKeys.Z_ERROR_FINAL] > z_mod) & (df[DatasetKeys.Z_ERROR_FINAL] <= z_grave),
            (df[DatasetKeys.Z_ERROR_FINAL] > z_leve) & (df[DatasetKeys.Z_ERROR_FINAL] <= z_mod),
            df[DatasetKeys.Z_ERROR_FINAL] < -z_grave,
            (df[DatasetKeys.Z_ERROR_FINAL] < -z_mod) & (df[DatasetKeys.Z_ERROR_FINAL] >= -z_grave),
            (df[DatasetKeys.Z_ERROR_FINAL] < -z_leve) & (df[DatasetKeys.Z_ERROR_FINAL] >= -z_mod)
        ]
        
        df[DatasetKeys.ALERTA_NIVEL] = np.select(condiciones, ModeloFisico.NIVEL_ALERTAS, default='Normal')
        
        # Limpieza de columnas intermedias temporales de pesos
        cols_to_drop = [f'z_{var}' for var in sospechosos.keys()] + columnas_pesos + ['suma_pesos']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Persistencia del checkpoint para auditoría
        logger.info(f"Registrando dataset intermedio Físicos en {Paths.PROC_CSV_AMAEM_FISICOS}")
        cols_to_save = [
            DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA, 
            DatasetKeys.PREDICCION_FOURIER, DatasetKeys.IMPACTO_EXOGENO, 
            DatasetKeys.RESIDUO, DatasetKeys.CONSUMO_FISICO_ESPERADO,
            DatasetKeys.Z_ERROR_FINAL, DatasetKeys.ALERTA_NIVEL
        ] + columnas_pct
            
        df[cols_to_save].drop_duplicates(subset=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA]).to_csv(Paths.PROC_CSV_AMAEM_FISICOS, index=False)
        
        logger.info("Enriquecimiento con variables físicas completado.")
        return df