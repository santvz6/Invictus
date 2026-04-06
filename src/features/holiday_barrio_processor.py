"""
Módulo de procesamiento de datos de festivos por barrio de Alicante.

Este componente se encarga de la ingesta y limpieza de los datos de festivos 
para enriquecer el dataset de consumo hídrico con variables de calendario local.
"""

import pandas as pd
from src.config import DatasetKeys, Paths, get_logger
logger = get_logger(__name__)

class HolidayBarrioProcessor:
    """
    Procesador especializado en la integración de festivos por barrio.
    """

    @staticmethod
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el enriquecimiento con datos de festivos.

        Args:
            df_amaem (pd.DataFrame): Dataset base de AMAEM.

        Returns:
            pd.DataFrame: Dataset enriquecido con información de festivos:
                - DIAS_FESTIVOS
                - PCT_FESTIVOS
                - ES_PUENTE  (binaria: ≥2 días festivos en el mes → efecto puente)
        """
        logger.info("Iniciando enriquecimiento con datos de festivos por barrio...")
        
        # 1. Preparación del dataset base
        df_final = HolidayBarrioProcessor._prepare_base_dataframe(df_amaem)

        # 2. Carga y limpieza de datos de festivos
        df_festivos = HolidayBarrioProcessor._load_and_clean_festivos_data()
        if df_festivos is None:
            df_final = df_final.drop(columns=['fecha_cruce_mensual'])
            return df_final

        # 3. Preparación de la referencia temporal en festivos
        df_festivos = HolidayBarrioProcessor._add_temporal_anchor(df_festivos)

        # 4. Integración de ambos datasets
        df_final = HolidayBarrioProcessor._execute_merge(df_final, df_festivos)

        # 5. Finalización: Imputación y guardado
        df_final = HolidayBarrioProcessor._finalize_data(df_final)

        logger.info("Enriquecimiento de festivos finalizado.")
        return df_final

    @staticmethod
    def _prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara el dataset de AMAEM creando el anclaje mensual para el cruce."""
        df_final = df.copy()
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')
        return df_final

    @staticmethod
    def _load_and_clean_festivos_data() -> pd.DataFrame | None:
        """Carga la fuente externa y normaliza cabeceras."""
        ruta_festivos = Paths.RAW_CSV_FESTIVOS
        try:
            df_festivos = pd.read_csv(ruta_festivos)
        except FileNotFoundError:
            logger.error(f"Archivo de festivos no localizado en {ruta_festivos}.")
            return None

        # Normalización de cabeceras
        df_festivos.columns = df_festivos.columns.str.strip()

        # Mapeo según el esquema de claves
        df_festivos = df_festivos.rename(columns={
            'Barrio': DatasetKeys.BARRIO,
            'Fecha': DatasetKeys.FECHA,
            'Dias_Festivos': DatasetKeys.DIAS_FESTIVOS,
            'Porcentaje_Anual': DatasetKeys.PCT_FESTIVOS
        })
            
        # Limpieza de porcentajes (eliminar '%' y convertir a float)
        if DatasetKeys.PCT_FESTIVOS in df_festivos.columns:
            df_festivos[DatasetKeys.PCT_FESTIVOS] = df_festivos[DatasetKeys.PCT_FESTIVOS].str.replace('%', '', regex=False).astype(float)
        
        return df_festivos

    @staticmethod
    def _add_temporal_anchor(df_festivos: pd.DataFrame) -> pd.DataFrame:
        """Genera el anclaje temporal en los datos de festivos."""
        if DatasetKeys.FECHA in df_festivos.columns:
            # La fecha viene como YYYY/MM en el CSV
            df_festivos['fecha_cruce_mensual'] = pd.to_datetime(df_festivos[DatasetKeys.FECHA], format='%Y/%m').dt.to_period('M')
            df_festivos = df_festivos.drop(columns=[DatasetKeys.FECHA])
        return df_festivos

    @staticmethod
    def _execute_merge(df_final: pd.DataFrame, df_festivos: pd.DataFrame) -> pd.DataFrame:
        """Realiza la integración de datos."""
        if DatasetKeys.BARRIO in df_festivos.columns:
            df_festivos[DatasetKeys.BARRIO] = df_festivos[DatasetKeys.BARRIO].astype(str).str.upper()
            df_final[DatasetKeys.BARRIO] = df_final[DatasetKeys.BARRIO].astype(str).str.upper()
            
            df_final = pd.merge(
                df_final, df_festivos,
                on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'],
                how='left'
            )
        return df_final

    @staticmethod
    def _finalize_data(df_final: pd.DataFrame) -> pd.DataFrame:
        """Limpia columnas técnicas, imputa nulos, añade ES_PUENTE y persiste el punto de control."""
        df_final = df_final.drop(columns=['fecha_cruce_mensual'])

        cols_festivos = [DatasetKeys.DIAS_FESTIVOS, DatasetKeys.PCT_FESTIVOS]
        for col in cols_festivos:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)  # Si no hay dato, asumimos 0 festivos

        # MEJORA 2: Feature de Puente
        # Un mes con ≥2 días festivos tiene alto riesgo de 'puente' (el efecto sobre el consumo
        # es diferente al de días festivos aislados).
        if DatasetKeys.DIAS_FESTIVOS in df_final.columns:
            df_final[DatasetKeys.ES_PUENTE] = (df_final[DatasetKeys.DIAS_FESTIVOS] >= 2).astype(int)
        else:
            df_final[DatasetKeys.ES_PUENTE] = 0

        # Registro del checkpoint
        logger.info(f"Registrando dataset intermedio de festivos en {Paths.PROC_CSV_STEP_FESTIVOS}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + [
            c for c in [DatasetKeys.DIAS_FESTIVOS, DatasetKeys.PCT_FESTIVOS, DatasetKeys.ES_PUENTE]
            if c in df_final.columns
        ]
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP_FESTIVOS, index=False)

        return df_final
