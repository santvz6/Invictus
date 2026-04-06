"""
Módulo de cálculo de consumo físico esperado mediante modelado híbrido.

Este componente combina un motor matemático basado en series de Fourier para 
capturar la estacionalidad física natural del agua, con un modelo de Machine 
Learning (Random Forest) que analiza el impacto de factores externos (clima, 
turismo, etc.) sobre el residuo estacional.
"""

import numpy as np
import pandas as pd
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
        logger.info("Iniciando cálculo de consumo físico esperado (Fourier Neutral + ML)...")

        # 1. Preparación y ordenación cronológica
        df = ModeloFisico._prepare_data(df)

        # 2. MEJORA 4: Añadir binarias estacionales (ortogonales a Fourier)
        df = ModeloFisico._add_seasonal_features(df)

        # 3. Fase de Fourier: Estacionalidad base por segmento [Barrio x Uso]
        #    MEJORA 3: Ajuste sólo sobre meses 'neutrales' (baja presión turística + sin festivos)
        df = ModeloFisico._calculate_fourier_neutral_baseline(df)

        # 4. Fase de ML: Modelado del impacto de variables exógenas
        df, rf_model, features_rf, X_full = ModeloFisico._calculate_ml_impact(df, feature_names)

        # 5. Consolidación y persistencia
        df = ModeloFisico._finalize_fisicos(df, rf_model, X_full)

        return df, rf_model, features_rf

    @staticmethod
    def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """Garantiza la integridad temporal necesaria para el ajuste de curvas."""
        df_copy = df.copy()
        df_copy[DatasetKeys.FECHA] = pd.to_datetime(df_copy[DatasetKeys.FECHA])
        return df_copy.sort_values(by=[DatasetKeys.BARRIO, DatasetKeys.USO, DatasetKeys.FECHA])

    @staticmethod
    def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        MEJORA 4: Añade features binarias estacionales ortogonales a Fourier.

        Estas variables son binarias (no sinusoidales), por lo que el RF puede aprender
        sus efectos específicos sin interferir con la onda de Fourier.
        Al contrario que Fourier, distinguen el tipo de evento (Semana Santa vs verano)
        y permiten capturar su impacto diferencial en el consumo.
        """
        df = df.copy()
        mes = pd.to_datetime(df[DatasetKeys.FECHA]).dt.month
        df[DatasetKeys.SEMANA_SANTA] = mes.isin([3, 4]).astype(int)
        df[DatasetKeys.VERANO]       = mes.isin([6, 7, 8]).astype(int)
        df[DatasetKeys.NAVIDAD]      = mes.isin([12, 1]).astype(int)
        logger.info("Features estacionales binarias añadidas: semana_santa, verano, navidad")
        return df

    @staticmethod
    def _calculate_fourier_neutral_baseline(df: pd.DataFrame) -> pd.DataFrame:
        """
        MEJORA 3 - Opción B: Fourier ajustado sólo sobre meses 'neutros'.

        Meses neutros = baja presión turística (pernoctaciones < mediana) Y sin festivos significativos.
        Esto evita que Fourier 'robe' la señal del turismo de verano y Semana Santa,
        dejando al RF espacio para aprender esos efectos en el residuo.

        Si un segmento no tiene suficientes meses neutros (≤6), cae al ajuste completo como fallback
        (equivalente al comportamiento anterior).
        """
        df[DatasetKeys.PREDICCION_FOURIER] = 0.0

        # Umbral de turismo: mediana global de pernoctaciones (si la columna existe)
        col_pernoct = DatasetKeys.PERNOCT_VT_PROV_INE
        if col_pernoct in df.columns:
            mediana_turismo = df[col_pernoct].median()
        else:
            mediana_turismo = None

        n_neutral = 0
        n_fallback = 0

        for (barrio, uso), group in df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO]):
            mask = (df[DatasetKeys.BARRIO] == barrio) & (df[DatasetKeys.USO] == uso)

            y_target_all = group[DatasetKeys.CONSUMO_RATIO].values
            t_arr_all    = np.arange(len(y_target_all))

            # Identificar meses neutros para el ajuste de Fourier
            if mediana_turismo is not None and DatasetKeys.DIAS_FESTIVOS in group.columns:
                mask_neutro = (
                    (group[col_pernoct] < mediana_turismo) &
                    (group[DatasetKeys.DIAS_FESTIVOS] == 0)
                ).values
            else:
                # Sin datos de turismo: usar solo el filtro de festivos
                mask_neutro = (group[DatasetKeys.DIAS_FESTIVOS] == 0).values if DatasetKeys.DIAS_FESTIVOS in group.columns else np.ones(len(group), dtype=bool)

            # Evitar Data Leakage: usar solo datos de 2022-2023 para el ajuste
            train_mask = (group[DatasetKeys.FECHA].dt.year <= 2023).values
            mask_neutro_train = mask_neutro & train_mask

            # Úsamos meses neutros si hay suficientes (≥6)
            use_neutral = mask_neutro_train.sum() >= 6

            if use_neutral:
                y_fit = y_target_all[mask_neutro_train]
                t_fit = t_arr_all[mask_neutro_train]
                n_neutral += 1
            else:
                # Fallback: ajuste completo con datos de 2022-2023
                y_fit = y_target_all[train_mask]
                t_fit = t_arr_all[train_mask]
                n_fallback += 1

            try:
                if len(y_fit) > 0:
                    coef, _ = curve_fit(
                        ModeloFisico._modelo_fourier, t_fit, y_fit,
                        p0=[0, np.mean(y_fit), 1000, 1000, 100, 100],
                        maxfev=10000
                    )
                else:
                    raise ValueError("Sin datos de entrenamiento")
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = ModeloFisico._modelo_fourier(t_arr_all, *coef)
            except Exception:
                logger.warning(f"Fallo en ajuste Fourier para {barrio} - {uso}. Aplicando media.")
                df.loc[mask, DatasetKeys.PREDICCION_FOURIER] = np.mean(y_fit) if len(y_fit) > 0 else np.mean(y_target_all)

        logger.info(f"Fourier NEUTRAL completado: {n_neutral} segmentos con ajuste neutral, {n_fallback} con fallback completo.")
        return df

    @staticmethod
    def _calculate_ml_impact(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, RandomForestRegressor, list[str], pd.DataFrame]:
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
        return df, ml_model, features_rf, X

    @staticmethod
    def _finalize_fisicos(df: pd.DataFrame, rf_model: RandomForestRegressor, X_full: pd.DataFrame) -> pd.DataFrame:
        """Combina componentes y genera los residuos finales para la detección de fraude y porcentajes de causa vía SHAP."""
        import shap
        
        # Híbrido: Base Física + Impacto de Contexto
        df[DatasetKeys.CONSUMO_FISICO_ESPERADO] = df[DatasetKeys.PREDICCION_FOURIER] + df[DatasetKeys.IMPACTO_EXOGENO]
        
        # Residuo Final: Diferencia entre consumo real y lo que la 'física' y el 'contexto' dictan
        # Un residuo positivo elevado es un fuerte indicador de posible fraude/fuga
        df[DatasetKeys.RESIDUO] = df[DatasetKeys.CONSUMO_RATIO] - df[DatasetKeys.CONSUMO_FISICO_ESPERADO]
        
        # Función auxiliar para cálculo robusto de Z-Score evitando avisos de división por cero
        def robust_zscore(x):
            std = x.std()
            if std == 0 or np.isnan(std):
                return 0.0
            return (x - x.mean()) / std

        df[DatasetKeys.Z_ERROR_FINAL] = df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO])[DatasetKeys.RESIDUO].transform(robust_zscore)
        
        # --- 4. Cálculo de Causas e Imputación AI (SHAP) ---
        # MEJORA: múltiples features pueden apuntar a la misma causa (CAUSAS_EXOGENAS).
        # Se ACUMULAN sus |SHAP| values en la misma causa (en lugar de sobrescribir).
        sospechosos_posibles = FeatureConfig.CAUSAS_EXOGENAS

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_full)
        feature_cols = X_full.columns.tolist()

        # Sigma local por barrio+uso para escalar la causa desconocida
        df['_sigma_local'] = df.groupby([DatasetKeys.BARRIO, DatasetKeys.USO])[DatasetKeys.RESIDUO].transform('std').fillna(1).replace(0, 1)

        df['_total_peso'] = 0.0

        # 1. Inicializar todas las columnas de causa a 0 (sin duplicados)
        columnas_pct_set = list(dict.fromkeys(sospechosos_posibles.values()))  # orden de inserción, sin duplicados
        for col_name in columnas_pct_set:
            df[f'_shap_{col_name}'] = 0.0

        # 2. Acumular |SHAP| por cada feature en su grupo de causa
        for var, col_name in sospechosos_posibles.items():
            if var in feature_cols:
                idx = feature_cols.index(var)
                # |SHAP| ya está en escala m³/cto → normalizar por sigma local
                shap_var = np.abs(shap_values[:, idx]) / df['_sigma_local']
            else:
                shap_var = 0.0
            df[f'_shap_{col_name}'] += shap_var
            df['_total_peso'] += shap_var if not isinstance(shap_var, float) else pd.Series([shap_var] * len(df), index=df.index)

        # 3. La causa desconocida = |z_error_final| (ya es adimensional)
        df['_peso_Desconocida'] = df[DatasetKeys.Z_ERROR_FINAL].abs()
        df['_total_peso'] += df['_peso_Desconocida']

        # Evitar división por cero
        df['_total_peso'] = df['_total_peso'].replace(0, 1)

        # 4. Convertir a porcentajes
        columnas_pct = []
        for col_name in columnas_pct_set:
            df[col_name] = (df[f'_shap_{col_name}'] / df['_total_peso']) * 100
            columnas_pct.append(col_name)

        df[DatasetKeys.PCT_CAUSA_DESCONOCIDA] = (df['_peso_Desconocida'] / df['_total_peso']) * 100
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
        cols_to_drop = [f'shap_{var}' for var in sospechosos_posibles.keys()] + ['_peso_Desconocida', '_total_peso', '_sigma_local']
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
        
        # Exportación a carpeta /riesgos/
        try:
            Paths.PROC_CSV_RIESGOS_DIR.mkdir(exist_ok=True, parents=True)
            for alerta_val in ModeloFisico.NIVEL_ALERTAS:
                df_alerta = df[df[DatasetKeys.ALERTA_NIVEL] == alerta_val].copy()
                
                # Para redondear los porcentajes de visualización en el archivo
                for c_pct in columnas_pct:
                    df_alerta[c_pct] = df_alerta[c_pct].round(1).astype(str) + '%'
                    
                # Ordenamiento de mayor a menor gravedad (o viceversa para defectos)
                ascending = 'DEFECTO' in alerta_val 
                df_alerta = df_alerta.sort_values(DatasetKeys.Z_ERROR_FINAL, ascending=ascending)
                
                alerta_filename = f"{alerta_val}.csv"
                df_alerta[cols_to_save].to_csv(Paths.PROC_CSV_RIESGOS_DIR / alerta_filename, index=False)
            logger.info(f"Reportes de alerta de riesgos exportados en {Paths.PROC_CSV_RIESGOS_DIR}")
        except Exception as e:
            logger.error(f"Error generando exportación estratificada de alertas: {e}")
        
        logger.info("Enriquecimiento con variables físicas completado.")
        return df