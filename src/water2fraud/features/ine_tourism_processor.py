import pandas as pd
import numpy as np
import logging

from src.config import DatasetKeys, Paths

logger = logging.getLogger(__name__)

class INETourismProcessor:
    """
    Clase encargada de procesar, limpiar y cruzar los datos externos de turismo 
    del Instituto Nacional de Estadística (INE) con los datos de AMAEM.
    """

    @staticmethod
    def enrich_with_tourism_data(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """
        Punto de entrada principal. Recibe el dataframe limpio de AMAEM y le inyecta 
        las variables turísticas del INE (Municipios y Provincia).
        """
        logger.info("Iniciando enriquecimiento con datos turísticos del INE...")
        
        # Guardamos el formato original de la fecha de AMAEM para restaurarlo al final
        formato_fecha_original = df_amaem[DatasetKeys.FECHA].copy()
        df_amaem[DatasetKeys.FECHA] = pd.to_datetime(df_amaem[DatasetKeys.FECHA]).dt.to_period('M')

        # 1. Procesar Municipios (Viviendas Turísticas interpoladas por Barrio)
        df_municipios = INETourismProcessor._process_municipios(df_amaem)
        
        # 2. Procesar Provincia (Ocupaciones y Pernoctaciones)
        df_provincia = INETourismProcessor._process_provincia()

        # 3. Merge Final
        logger.info("Cruzando datos de AMAEM con INE Municipios y Provincia...")
        df_final = pd.merge(df_amaem, df_municipios, on=[DatasetKeys.BARRIO, DatasetKeys.FECHA], how='left')
        df_final = pd.merge(df_final, df_provincia, on=DatasetKeys.FECHA, how='left')

        # Rellenar posibles nulos generados por el cruce con 0
        cols_ine = [DatasetKeys.NUM_VT_BARRIO, DatasetKeys.PCT_TURISTICO_REAL, DatasetKeys.OCUPACIONES_VT, DatasetKeys.PERNOCTACIONES_VT]
        for col in cols_ine:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        # Restaurar formato de fecha original
        df_final[DatasetKeys.FECHA] = formato_fecha_original
        
        logger.info("Enriquecimiento con INE completado con éxito.")
        return df_final

    @staticmethod
    def _process_municipios(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """Procesa datos de viviendas turísticas por municipio y los mapea a barrios."""
        
        # 1. MAPPING
        df_mun = pd.read_csv(Paths.INE_MUNICIPIOS_PLAZAS, encoding="latin1", sep="\t")
        df_mapping = pd.read_csv(Paths.MAPPING_BARRIOS, sep=";")
        
        # Limpieza de fechas y números
        df_mun[DatasetKeys.FECHA] = pd.to_datetime(df_mun['Periodo'].str.replace('M', '-')).dt.to_period('M')
        if 'Total' in df_mun.columns:
            df_mun['Total'] = df_mun['Total'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_mun['Total'] = pd.to_numeric(df_mun['Total'], errors='coerce').fillna(0)

        # Filtramos solo viviendas
        df_mun = df_mun[df_mun['Viviendas y plazas'].str.contains('Viviendas', na=False, case=False)]
        df_mun_data = df_mun[['Municipios', DatasetKeys.FECHA, 'Total']].rename(columns={'Total': 'Total_vt_municipio'})

        # Cruzamos con los pesos de los barrios
        df_weighted = pd.merge(df_mapping, df_mun_data, left_on='municipio', right_on='Municipios')
        df_weighted['peso'] = pd.to_numeric(df_weighted['peso'], errors='coerce').fillna(0)
        df_weighted['Total_vt_municipio'] = pd.to_numeric(df_weighted['Total_vt_municipio'], errors='coerce').fillna(0)
        df_weighted[DatasetKeys.NUM_VT_BARRIO] = df_weighted['Total_vt_municipio'] * df_weighted['peso']

        df_barrio = df_weighted.groupby([DatasetKeys.BARRIO, DatasetKeys.FECHA])[DatasetKeys.NUM_VT_BARRIO].sum().reset_index()

        # 2. PORCENTAJE VIVIENDAS TURÍSTICAS
        # Extraemos el número de contratos de AMAEM (solo doméstico) para calcular el porcentaje
        df_amaem_dom = df_amaem[df_amaem[DatasetKeys.USO] == 'DOMESTICO'][[DatasetKeys.BARRIO, DatasetKeys.FECHA, DatasetKeys.NUM_CONTRATOS]]
        df_merge = pd.merge(df_barrio, df_amaem_dom, on=[DatasetKeys.BARRIO, DatasetKeys.FECHA], how='left')

        # 3. INTERPOLACIÓN MENSUAL LINEAL
        periodo_minimo = df_merge[DatasetKeys.FECHA].min()
        rango_completo = pd.period_range(start=periodo_minimo, end='2024-12', freq='M')

        def interpolate_group(group):
            group = group.set_index(DatasetKeys.FECHA).reindex(rango_completo)
            if DatasetKeys.NUM_VT_BARRIO in group.columns:
                group[DatasetKeys.NUM_VT_BARRIO] = group[DatasetKeys.NUM_VT_BARRIO].interpolate(method='linear')
            return group.loc['2022-01':'2024-12']

        df_interpolated = (df_merge[[DatasetKeys.BARRIO, DatasetKeys.FECHA, DatasetKeys.NUM_VT_BARRIO]]
                           .groupby(DatasetKeys.BARRIO, group_keys=True)
                           .apply(interpolate_group, include_groups=False)
                           .reset_index()
                           .rename(columns={'level_1': DatasetKeys.FECHA}))

        # Merge final con contratos para el porcentaje interpolado
        df_resampled = pd.merge(df_interpolated, df_amaem_dom, on=[DatasetKeys.BARRIO, DatasetKeys.FECHA], how='left')
        df_resampled[DatasetKeys.PCT_TURISTICO_REAL] = (df_resampled[DatasetKeys.NUM_VT_BARRIO] / df_resampled[DatasetKeys.NUM_CONTRATOS]) * 100
        df_resampled[DatasetKeys.PCT_TURISTICO_REAL] = df_resampled[DatasetKeys.PCT_TURISTICO_REAL].fillna(0).round(4)

        return df_resampled[[DatasetKeys.BARRIO, DatasetKeys.FECHA, DatasetKeys.NUM_VT_BARRIO, DatasetKeys.PCT_TURISTICO_REAL]]

    @staticmethod
    def _process_provincia() -> pd.DataFrame:
        """Procesa datos de ocupaciones y pernoctaciones a nivel provincial."""
        df_prov = pd.read_csv(Paths.INE_PROVINCIA_VT, encoding="utf-8", sep=";")
        
        # Limpieza básica de nombres de columnas
        df_prov.columns = (df_prov.columns
                           .str.strip().str.lower()
                           .str.replace(' ', '_').str.replace(':', '')
                           .str.replace('á', 'a').str.replace('é', 'e')
                           .str.replace('í', 'i').str.replace('ó', 'o')
                           .str.replace('ú', 'u'))
        
        df_prov = df_prov.rename(columns={
            "fecha": "fecha_orig",
            "total_numero_de_alojamientos_turisticos_ocupados": DatasetKeys.OCUPACIONES_VT,
            "numero_de_noches_ocupadas": DatasetKeys.PERNOCTACIONES_VT
        })

        df_prov[DatasetKeys.FECHA] = pd.to_datetime(df_prov['fecha_orig'].str.replace('M', '-')).dt.to_period('M')
        df_prov_clean = df_prov[[DatasetKeys.FECHA, DatasetKeys.OCUPACIONES_VT, DatasetKeys.PERNOCTACIONES_VT]].copy()

        # Limpieza de puntos en miles
        for col in [DatasetKeys.OCUPACIONES_VT, DatasetKeys.PERNOCTACIONES_VT]:
            if df_prov_clean[col].dtype == 'object':
                df_prov_clean[col] = df_prov_clean[col].str.replace('.', '', regex=False).astype(float)
            elif df_prov_clean[col].dtype in ['float64', 'int64']:
                df_prov_clean[col] = (df_prov_clean[col] * 1000).astype(int) # Según tu lógica del notebook

        return df_prov_clean