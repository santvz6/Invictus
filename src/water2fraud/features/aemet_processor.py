import pandas as pd
import logging

from src.config import DatasetKeys, Paths

logger = logging.getLogger(__name__)

class AEMETProcessor:
    """
    Clase encargada de procesar, limpiar y cruzar los datos externos climáticos 
    de AEMET con los datos base de AMAEM.
    """

    @staticmethod
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando enriquecimiento con datos climáticos de AEMET...")
        
        # 0. Crear copia y ancla mensual para el cruce (igual que en INE)
        df_final = df_amaem.copy()
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')

        # 1. Cargar datos de AEMET
        ruta_aemet = Paths.AEMET_CLIMA_BARRIOS
        
        try:
            df_aemet = pd.read_csv(ruta_aemet)
        except FileNotFoundError:
            logger.error(f"No se encontró el archivo AEMET en {ruta_aemet}. Omitiendo cruce climático.")
            df_final = df_final.drop(columns=['fecha_cruce_mensual'])
            return df_final

        # 2. Limpieza de columnas del CSV de AEMET
        df_aemet.columns = df_aemet.columns.str.strip().str.lower().str.replace(' ', '_')

        # Renombramos las columnas usando el esquema específico de tu dataset (zona, tm_mes, p_mes)
        df_aemet = df_aemet.rename(columns={
            'zona': DatasetKeys.BARRIO,
            'tm_mes': DatasetKeys.TEMP_MEDIA,
            'p_mes': DatasetKeys.PRECIPITACION
        })
            
        # Limpieza de valores (ej. '23,5' -> 23.5) por si vienen como texto
        for col in [DatasetKeys.TEMP_MEDIA, DatasetKeys.PRECIPITACION]:
            if col in df_aemet.columns and df_aemet[col].dtype == object:
                df_aemet[col] = df_aemet[col].str.replace(',', '.', regex=False).astype(float)

        # Crear ancla mensual en los datos de AEMET
        if 'fecha' in df_aemet.columns:
            df_aemet['fecha'] = pd.to_datetime(df_aemet['fecha'])
            df_aemet['fecha_cruce_mensual'] = df_aemet['fecha'].dt.to_period('M')
            # Eliminamos la fecha del CSV de AEMET antes de cruzar para evitar sufijos o colisiones
            df_aemet = df_aemet.drop(columns=['fecha'])
        else:
            logger.warning("No se encontró la columna de fecha en AEMET.")

        # 3. Cruzar datos (Merge)
        if DatasetKeys.BARRIO in df_aemet.columns:
            # Normalizamos el barrio a mayúsculas para asegurar el emparejamiento con AMAEM
            df_aemet[DatasetKeys.BARRIO] = df_aemet[DatasetKeys.BARRIO].astype(str).str.upper()
            
            df_final = pd.merge(
                df_final,
                df_aemet,
                on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'],
                how='left'
            )
        elif 'fecha_cruce_mensual' in df_aemet.columns:
            # Fallback: Si el clima es general para todo Alicante (no hay columna barrio)
            df_final = pd.merge(df_final, df_aemet, on='fecha_cruce_mensual', how='left')

        # 4. Limpieza Final
        df_final = df_final.drop(columns=['fecha_cruce_mensual'])

        # Rellenar posibles valores nulos generados por barrios sin estación meteorológica
        cols_clima = [DatasetKeys.TEMP_MEDIA, DatasetKeys.PRECIPITACION] 
        for col in cols_clima:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(df_final[col].mean())
                
        logger.info(f"Guardando dataset intermedio en {Paths.PROC_CSV_STEP_AEMET}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + [c for c in cols_clima if c in df_final.columns]
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP_AEMET, index=False)

        logger.info("Enriquecimiento con AEMET completado con éxito.")
        return df_final