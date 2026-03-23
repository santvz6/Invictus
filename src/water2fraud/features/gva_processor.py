"""
Módulo de procesamiento de datos turísticos oficiales de la Generalitat Valenciana (GVA).

Este componente analiza los registros históricos de Viviendas Turísticas y Hoteles, 
calculando el volumen de oferta activa para cada periodo mensual mediante la 
evaluación de los estados de alta y baja administrativa de cada establecimiento.
"""

import pandas as pd
import logging

from src.config import DatasetKeys, Paths

# Configuración del logger para el seguimiento de la carga turística
logger = logging.getLogger(__name__)

class GVAProcessor:
    """
    Procesador de registros administrativos de turismo de la Comunidad Valenciana.
    
    Especializado en transformar microdatos de registros individuales en series 
    temporales agregadas, considerando la vigencia temporal de cada licencia 
    (fechas de apertura y clausura).
    """

    @staticmethod
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline de enriquecimiento con datos de la GVA.

        Analiza tanto viviendas turísticas como establecimientos hoteleros para 
        determinar la presión turística legal en cada barrio y mes.

        Args:
            df_amaem (pd.DataFrame): Dataset base con registros de consumo.

        Returns:
            pd.DataFrame: Dataset enriquecido con métricas de oferta turística.
        """
        logger.info("Iniciando enriquecimiento con datos turísticos de la GVA (Altas y Bajas)...")
        
        # 1. Preparación del marco temporal base
        df_final = GVAProcessor._prepare_base_dataframe(df_amaem)
        meses_unicos = df_final['fecha_cruce_mensual'].unique()

        # 2. Obtención de oferta activa por categoría
        df_vt, df_hoteles = GVAProcessor._get_tourist_activity(meses_unicos)

        # 3. Integración de la oferta turística registrada
        df_final = GVAProcessor._merge_tourist_data(df_final, df_vt, df_hoteles)

        # 4. Finalización y guardado de punto de control
        df_final = GVAProcessor._finalize_gva(df_final)

        return df_final

    @staticmethod
    def _prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Sincroniza el formato de fecha y genera el ancla mensual para el cruce."""
        df_final = df.copy()
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')
        return df_final

    @staticmethod
    def _get_tourist_activity(meses_unicos: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Orquesta el procesamiento individual de cada fuente de datos de la GVA."""
        # Procesamiento de Viviendas Turísticas (VT)
        df_vt = GVAProcessor._process_gva_source(
            filepath=Paths.GVA_VIVIENDAS, 
            meses_unicos=meses_unicos, 
            prefix='viviendas'
        )

        # Procesamiento de Hoteles y establecimientos similares
        df_hoteles = GVAProcessor._process_gva_source(
            filepath=Paths.GVA_HOTELES, 
            meses_unicos=meses_unicos, 
            prefix='hoteles'
        )
        return df_vt, df_hoteles

    @staticmethod
    def _process_gva_source(filepath, meses_unicos, prefix) -> pd.DataFrame:
        """
        Calcula la oferta activa por mes evaluando fechas de alta y baja.
        
        A diferencia de otros procesadores, este evalúa cada registro unitario contra 
        todos los meses del histórico para determinar su vigencia funcional.
        """
        try:
            # Lectura con codificación específica para caracteres regionales
            df = pd.read_csv(filepath, sep=';', encoding='latin1', on_bad_lines='skip')
        except FileNotFoundError:
            logger.error(f"Fuente GVA no encontrada en {filepath}. Se omite el bloque {prefix}.")
            return pd.DataFrame()

        # Estandarización de cabeceras
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Mapeo de columnas críticas del registro GVA
        col_alta = 'fecha_alta'
        col_baja = 'fecha_baja'
        col_plazas = 'plazas'

        # Limpieza y tipado de fechas (manejo del formato regional dd/mm/yyyy)
        if col_alta in df.columns:
            df[col_alta] = pd.to_datetime(df[col_alta], format='%d/%m/%Y', dayfirst=True, errors='coerce')
        else:
            return pd.DataFrame()
            
        df[col_baja] = pd.to_datetime(df[col_baja], format='%d/%m/%Y', dayfirst=True, errors='coerce') if col_baja in df.columns else pd.NaT

        # Validación de plazas alojativas
        df[col_plazas] = pd.to_numeric(df[col_plazas], errors='coerce').fillna(0) if col_plazas in df.columns else 0

        # Identificación de multiplicadores por bloque (Nm. Apartamentos)
        col_apt = next((c for c in df.columns if 'apartamentos' in c), None)
        df['peso_unidad'] = pd.to_numeric(df[col_apt], errors='coerce').fillna(1) if col_apt else 1

        # Cálculo de agregados mensuales
        results = []
        for period in meses_unicos:
            month_start = period.to_timestamp(how='start')
            month_end = period.to_timestamp(how='end')

            # Criterio de establecimiento activo: 
            # Registrado antes del fin de mes y sin fecha de baja previa al inicio de mes
            cond_alta = df[col_alta].notna() & (df[col_alta] <= month_end)
            cond_baja = df[col_baja].isna() | (df[col_baja] >= month_start)

            active_df = df[cond_alta & cond_baja]
            
            results.append({
                'fecha_cruce_mensual': period,
                f'num_{prefix}_gva': active_df['peso_unidad'].sum(),
                f'plazas_{prefix}_gva': active_df[col_plazas].sum()
            })

        return pd.DataFrame(results)

    @staticmethod
    def _merge_tourist_data(df_final: pd.DataFrame, df_vt: pd.DataFrame, df_hoteles: pd.DataFrame) -> pd.DataFrame:
        """Integra las series temporales de oferta turística en el dataset principal."""
        # Integración de Viviendas Turísticas
        if not df_vt.empty:
            df_vt = df_vt.rename(columns={
                'num_viviendas_gva': DatasetKeys.NUM_VT_BARRIO_GVA,
                'plazas_viviendas_gva': DatasetKeys.PLAZAS_VIVIENDAS_GVA
            })
            df_final = pd.merge(df_final, df_vt, on='fecha_cruce_mensual', how='left')
        else:
            df_final[DatasetKeys.NUM_VT_BARRIO_GVA] = 0
            df_final[DatasetKeys.PLAZAS_VIVIENDAS_GVA] = 0

        # Integración de Hoteles
        if not df_hoteles.empty:
            df_hoteles = df_hoteles.rename(columns={
                'num_hoteles_gva': DatasetKeys.NUM_HOTELES_BARRIO_GVA,
                'plazas_hoteles_gva': DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA
            })
            df_final = pd.merge(df_final, df_hoteles, on='fecha_cruce_mensual', how='left')
        else:
            df_final[DatasetKeys.NUM_HOTELES_BARRIO_GVA] = 0
            df_final[DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA] = 0
            
        return df_final

    @staticmethod
    def _finalize_gva(df_final: pd.DataFrame) -> pd.DataFrame:
        """Limpia variables temporales, gestiona nulos y guarda el histórico intermedio."""
        cols_gva = [
            DatasetKeys.NUM_VT_BARRIO_GVA, 
            DatasetKeys.PLAZAS_VIVIENDAS_GVA, 
            DatasetKeys.NUM_HOTELES_BARRIO_GVA, 
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA
        ]
        
        # Imputación de nulos (meses sin cobertura GVA se asumen con oferta cero)
        for col in cols_gva:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        df_final = df_final.drop(columns=['fecha_cruce_mensual'])

        # Registro del punto de control
        logger.info(f"Registrando dataset intermedio GVA en {Paths.PROC_CSV_STEP_GVA}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + [c for c in cols_gva if c in df_final.columns]
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP_GVA, index=False)

        return df_final