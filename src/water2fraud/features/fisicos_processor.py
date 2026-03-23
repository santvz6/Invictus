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
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor

from src.config import DatasetKeys, Paths

# Configuración del logger para seguimiento del modelado físico
logger = logging.getLogger(__name__)

class FisicosProcessor:
    """
    Procesador de modelado híbrido para la determinación del consumo base.
    
    Utiliza una aproximación de dos etapas:
    1. Base Física: Onda estacional de Fourier ajustada por Barrio y Uso.
    2. Componente de Impacto: Modelado de residuos mediante factores exógenos.
    """

    @staticmethod
    def _modelo_fourier(t, m, c, a1, b1, a2, b2):
        """
        Ecuación de Fourier de segundo orden con componente tendencial.
        Define la 'huella dactilar' del consumo físico esperado bajo condiciones ideales.
        """
        w = 2 * np.pi / 12  # Frecuencia fundamental (anual)
        return (m * t + c) + (a1 * np.cos(w * t) + b1 * np.sin(w * t)) + (a2 * np.cos(2 * w * t) + b2 * np.sin(2 * w * t))

    @staticmethod
    def process(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """
        Ejecuta el pipeline de cálculo del consumo físico esperado.

        Args:
            df (pd.DataFrame): Dataset enriquecido con variables externas.
            feature_names (list[str]): Lista de columnas exógenas a considerar.

        Returns:
            pd.DataFrame: Dataset con columnas de predicción física y residuos de anomalía.
        """
        logger.info("Iniciando cálculo de consumo físico esperado (Fourier + ML)...")
        
        # 1. Preparación y ordenación cronológica
        df = FisicosProcessor._prepare_data(df)

        # 2. Fase de Fourier: Estacionalidad base por segmento [Barrio x Uso]
        df = FisicosProcessor._calculate_fourier_baseline(df)

        # 3. Fase de ML: Modelado del impacto de variables exógenas
        df = FisicosProcessor._calculate_ml_impact(df, feature_names)

        # 4. Consolidación y persistencia
        df = FisicosProcessor._finalize_fisicos(df)

        return df

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
            y_target = group[DatasetKeys.CONSUMO_RATIO].values
            t_arr = np.arange(len(y_target))
            
            try:
                # Ajuste no lineal de parámetros (m, c, a1, b1, a2, b2)
                coef, _ = curve_fit(
                    FisicosProcessor._modelo_fourier, t_arr, y_target, 
                    p0=[0, np.mean(y_target), 1000, 1000, 100, 100], maxfev=10000
                )
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = FisicosProcessor._modelo_fourier(t_arr, *coef)
            except Exception:
                # Fallback: Si el ajuste falla por falta de datos, se usa el valor medio histórico
                logger.warning(f"Fallo en ajuste Fourier para {barrio} - {uso}. Aplicando media.")
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = np.mean(y_target)
        
        return df

    @staticmethod
    def _calculate_ml_impact(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """Entrena un modelo para predecir cuánto del consumo depende de factores externos."""
        # Cálculo del residuo estacional (lo que Fourier no explica)
        df[DatasetKeys.RESIDUO] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.PREDICCION_FOURIER]

        # Limpieza de variables exógenas
        exogenas = [col for col in feature_names if col in df.columns and col != DatasetKeys.CONSUMO_RATIO]
        for col in exogenas:
            df[col] = df[col].fillna(df[col].mean())

        # Añadimos contexto de fecha (mes) y uso para mejorar la capacidad predictiva sin memorizar barrios
        df['mes_temp'] = pd.to_datetime(df[DatasetKeys.FECHA]).dt.month
        
        # Generamos contextos de uso (DOMESTICO vs OTROS) - es de baja cardinalidad, seguro contra leakage masivo
        df_ml = pd.get_dummies(df, columns=[DatasetKeys.USO])
        columnas_contexto = [col for col in df_ml.columns if col.startswith(DatasetKeys.USO + '_')]
        
        X = df_ml[exogenas + [DatasetKeys.PREDICCION_FOURIER, 'mes_temp'] + columnas_contexto]
        y = df_ml[DatasetKeys.RESIDUO]
        
        # Entrenamiento del modelo de impacto con restricción de profundidad para forzar generalización
        ml_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(X, y)
        
        df[DatasetKeys.IMPACTO_EXOGENO] = ml_model.predict(X)
        return df

    @staticmethod
    def _finalize_fisicos(df: pd.DataFrame) -> pd.DataFrame:
        """Combina componentes y genera los residuos finales para la detección de fraude."""
        # Híbrido: Base Física + Impacto de Contexto
        df[DatasetKeys.CONSUMO_FISICO_ESPERADO] = df[DatasetKeys.PREDICCION_FOURIER] + df[DatasetKeys.IMPACTO_EXOGENO]
        
        # Residuo Final: Diferencia entre consumo real y lo que la 'física' y el 'contexto' dictan
        # Un residuo positivo elevado es un fuerte indicador de posible fraude/fuga
        df[DatasetKeys.RESIDUO] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.CONSUMO_FISICO_ESPERADO]
        
        # Persistencia del checkpoint para auditoría
        logger.info(f"Registrando dataset intermedio Físicos en {Paths.PROC_CSV_AMAEM_FISICOS}")
        cols_to_save = [
            DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA, 
            DatasetKeys.PREDICCION_FOURIER, DatasetKeys.IMPACTO_EXOGENO, 
            DatasetKeys.RESIDUO, DatasetKeys.CONSUMO_FISICO_ESPERADO
        ]
            
        df[cols_to_save].drop_duplicates(subset=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA]).to_csv(Paths.PROC_CSV_AMAEM_FISICOS, index=False)
        
        logger.info("Enriquecimiento con variables físicas completado.")
        return df