import numpy as np
import pandas as pd
import logging
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor

from src.config import DatasetKeys, Paths

logger = logging.getLogger(__name__)

class FisicosProcessor:
    """
    Clase encargada de calcular el consumo físico esperado de agua mediante 
    un modelo predictivo híbrido: 
    Onda estacional base (Fourier) + Comportamiento de impacto (Machine Learning).
    """

    @staticmethod
    def _modelo_fourier(t, m, c, a1, b1, a2, b2):
        """Ecuación matemática para la estacionalidad física del agua."""
        w = 2 * np.pi / 12
        return (m * t + c) + (a1 * np.cos(w * t) + b1 * np.sin(w * t)) + (a2 * np.cos(2 * w * t) + b2 * np.sin(2 * w * t))

    @staticmethod
    def process(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        logger.info("Iniciando cálculo de consumo físico esperado (Fourier + ML)...")
        df = df.copy()

        # 1. Asegurar orden cronológico para el cálculo del tiempo 't'
        df[DatasetKeys.FECHA] = pd.to_datetime(df[DatasetKeys.FECHA])
        df = df.sort_values(by=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA])

        # 2. MOTOR MATEMÁTICO (FOURIER) INDEPENDIENTE POR [BARRIO x USO]
        df[DatasetKeys.PREDICCION_FOURIER] = 0.0

        for (barrio, uso), group in df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO]):
            mask = (df[DatasetKeys.BARRIO] == barrio) & (df[DatasetKeys.USO] == uso)
            y_target = group[DatasetKeys.CONSUMO_RATIO].values
            t_arr = np.arange(len(y_target))
            
            try:
                coef, _ = curve_fit(
                    FisicosProcessor._modelo_fourier, t_arr, y_target, 
                    p0=[0, np.mean(y_target), 1000, 1000, 100, 100], maxfev=10000
                )
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = FisicosProcessor._modelo_fourier(t_arr, *coef)
            except Exception:
                # Fallback estático (media) en caso de datos insuficientes o demasiado ruidosos
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = np.mean(y_target)

        # 3. MACHINE LEARNING (PREDICCIÓN DE COMPORTAMIENTO)
        df[DatasetKeys.RESIDUO] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.PREDICCION_FOURIER]

        exogenas = [col for col in feature_names if col in df.columns]
        for col in exogenas:
            df[col] = df[col].fillna(df[col].mean())

        df_ml = pd.get_dummies(df, columns=[DatasetKeys.BARRIO, DatasetKeys.USO])
        columnas_contexto = [col for col in df_ml.columns if col.startswith(DatasetKeys.BARRIO + '_') or col.startswith(DatasetKeys.USO + '_')]
        
        X, y = df_ml[exogenas + columnas_contexto], df_ml[DatasetKeys.RESIDUO]
        ml_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
        df[DatasetKeys.IMPACTO_EXOGENO] = ml_model.predict(X)

        # 4. HÍBRIDO FINAL
        df[DatasetKeys.CONSUMO_FISICO_ESPERADO] = df[DatasetKeys.PREDICCION_FOURIER] + df[DatasetKeys.IMPACTO_EXOGENO]
        
        logger.info(f"Guardando dataset intermedio en {Paths.PROC_CSV_AMAEM_FISICOS}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA, 
                        DatasetKeys.PREDICCION_FOURIER, DatasetKeys.IMPACTO_EXOGENO, 
                        DatasetKeys.RESIDUO, DatasetKeys.CONSUMO_FISICO_ESPERADO]
            
        df[cols_to_save].drop_duplicates(subset=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA]).to_csv(Paths.PROC_CSV_AMAEM_FISICOS, index=False)
        
        logger.info("Enriquecimiento con variables Físicas completado con éxito.")
        return df