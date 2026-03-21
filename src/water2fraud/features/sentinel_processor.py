import pandas as pd
import logging
from src.config import DatasetKeys, Paths

logger = logging.getLogger(__name__)

class SentinelProcessor:
    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando procesamiento de NDVI (Sentinel)...")
        
        ruta_ndvi = Paths.SENTINEL_NDVI
        
        if not ruta_ndvi.exists():
            logger.warning(f"No se encontró el archivo NDVI en {ruta_ndvi}. Se omite este paso.")
            return df
            
        try:
            df_ndvi = pd.read_csv(ruta_ndvi)
            
            # Anclaje mensual
            df['fecha_cruce_mensual'] = df[DatasetKeys.FECHA].dt.to_period('M')
            
            col_fecha = 'fecha' if 'fecha' in df_ndvi.columns else 'fecha_mes'
            df_ndvi['fecha_cruce_mensual'] = pd.to_datetime(df_ndvi[col_fecha]).dt.to_period('M')
            
            # Identificamos la columna barrio en el dataset de NDVI
            col_barrio = next((c for c in df_ndvi.columns if c.lower() in ['barrio', 'neighborhood']), None)
            
            merge_keys = ['fecha_cruce_mensual']
            
            if col_barrio:
                if col_barrio != DatasetKeys.BARRIO:
                    df_ndvi = df_ndvi.rename(columns={col_barrio: DatasetKeys.BARRIO})
                
                # Normalizamos el barrio (mayúsculas) para evitar fallos de matching
                df_ndvi[DatasetKeys.BARRIO] = df_ndvi[DatasetKeys.BARRIO].astype(str).str.upper()
                merge_keys.append(DatasetKeys.BARRIO)
            
            # Eliminamos posibles duplicados en origen antes del cruce (crucial para evitar multiplicaciones)
            df_ndvi = df_ndvi.drop_duplicates(subset=merge_keys)
            
            # Realizamos el Merge (ahora sí, por Barrio y Fecha)
            df = pd.merge(df, df_ndvi[merge_keys + ['ndvi_satelite']], on=merge_keys, how='left')
            df = df.rename(columns={'ndvi_satelite': DatasetKeys.NDVI_SATELITE})
            df = df.drop(columns=['fecha_cruce_mensual'])
            
            # Guardar el CSV del paso extrayendo solo 1 fila por combinación Barrio-Fecha
            ruta_csv = Paths.PROC_CSV_STEP5_SENTINEL
            logger.info(f"Guardando dataset intermedio en {ruta_csv}")
            
            cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA, DatasetKeys.NDVI_SATELITE]
            cols_to_save = [c for c in cols_to_save if c in df.columns]
            
            df[cols_to_save].drop_duplicates().to_csv(ruta_csv, index=False)
            
            logger.info("Enriquecimiento con Sentinel completado con éxito.")
        
        except Exception as e:
            logger.error(f"Error procesando datos de Sentinel: {e}")
            
        return df
