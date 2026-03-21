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
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando enriquecimiento con datos turísticos del INE...")
        
        # 0. Creamos una copia para no alterar el original y generamos un ancla mensual (Period)
        df_final = df_amaem.copy()
        # Aseguramos que la fecha es datetime antes de crear el ancla
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        
        # 'fecha_cruce_mensual' será ej: "2024-12" (tipo Period). Esto es vital para el join.
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')

        # 1. Procesar Municipios (Viviendas Turísticas interpoladas por Barrio)
        # Le pasamos el df_final porque necesitamos los datos domésticos
        df_municipios = INETourismProcessor._process_municipios(df_final)
        
        # 2. Procesar Provincia (Ocupaciones y Pernoctaciones)
        df_provincia = INETourismProcessor._process_provincia()

        # 3. Merge Final
        logger.info("Cruzando datos de AMAEM con INE Municipios y Provincia...")
        
        # Cruzamos Municipios usando el Barrio y nuestra NUEVA ancla mensual
        df_final = pd.merge(
            df_final, 
            df_municipios, 
            on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], 
            how='left'
        )
        
        # Cruzamos Provincia usando solo nuestra NUEVA ancla mensual
        df_final = pd.merge(
            df_final, 
            df_provincia, 
            on='fecha_cruce_mensual', 
            how='left'
        )

        # Rellenar posibles nulos generados por el cruce con 0
        cols_ine = [DatasetKeys.NUM_VT_BARRIO, DatasetKeys.PCT_VT_BARRIO, DatasetKeys.OCUPACIONES_VT_PROV, DatasetKeys.PERNOCTACIONES_VT_PROV]
        for col in cols_ine:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        # 4. Limpieza Final: Borramos el ancla temporal de cruce. 
        # ¡La columna original DatasetKeys.FECHA ha permanecido intacta todo el tiempo!
        df_final = df_final.drop(columns=['fecha_cruce_mensual'])
        
        logger.info(f"Guardando dataset intermedio en {Paths.PROC_CSV_STEP2_INE}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + [c for c in cols_ine if c in df_final.columns]
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP2_INE, index=False)
        
        logger.info("Enriquecimiento con INE completado con éxito.")
        return df_final

    ######################################################################################
    #                                        MUNICIPIOS
    @staticmethod
    def _map_mun2barrios():
        df_mun = pd.read_csv(Paths.INE_MUNICIPIOS_PLAZAS, encoding="latin1", sep="\t")
        df_mapping = pd.read_csv(Paths.MAPPING_BARRIOS, sep=";")
        
        # Limpieza de fechas: Creamos 'fecha_cruce_mensual'
        df_mun['fecha_cruce_mensual'] = pd.to_datetime(df_mun['Periodo'].str.replace('M', '-')).dt.to_period('M')
        
        if 'Total' in df_mun.columns:
            df_mun['Total'] = df_mun['Total'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_mun['Total'] = pd.to_numeric(df_mun['Total'], errors='coerce').fillna(0)

        df_mun = df_mun[df_mun['Viviendas y plazas'].str.contains('Viviendas', na=False, case=False)]
        df_mun_data = df_mun[['Municipios', 'fecha_cruce_mensual', 'Total']].rename(columns={'Total': 'Total_vt_municipio'})

        df_weighted = pd.merge(df_mapping, df_mun_data, left_on='municipio', right_on='Municipios')
        df_weighted['peso'] = pd.to_numeric(df_weighted['peso'], errors='coerce').fillna(0)
        df_weighted['Total_vt_municipio'] = pd.to_numeric(df_weighted['Total_vt_municipio'], errors='coerce').fillna(0)
        df_weighted[DatasetKeys.NUM_VT_BARRIO] = df_weighted['Total_vt_municipio'] * df_weighted['peso']

        df_barrio = df_weighted.groupby([DatasetKeys.BARRIO, 'fecha_cruce_mensual'])[DatasetKeys.NUM_VT_BARRIO].sum().reset_index()
        return df_barrio
    
    @staticmethod
    def _merge_domesticos_ine(df_amaem, df_barrio):
        df_amaem_dom = df_amaem[df_amaem[DatasetKeys.USO] == 'DOMESTICO'][[DatasetKeys.BARRIO, 'fecha_cruce_mensual', DatasetKeys.NUM_CONTRATOS]]
        df_merge = pd.merge(df_barrio, df_amaem_dom, on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], how='left')
        return df_amaem_dom, df_merge

    @staticmethod
    def _interpolacion_mensual(df_merge):
        periodo_minimo = df_merge['fecha_cruce_mensual'].min()
        rango_completo = pd.period_range(start=periodo_minimo, end='2024-12', freq='M')

        def interpolate_group(group):
            group = group.set_index('fecha_cruce_mensual').reindex(rango_completo)
            if DatasetKeys.NUM_VT_BARRIO in group.columns:
                group[DatasetKeys.NUM_VT_BARRIO] = group[DatasetKeys.NUM_VT_BARRIO].interpolate(method='linear')
            return group.loc['2022-01':'2024-12']

        df_interpolated = (df_merge[[DatasetKeys.BARRIO, 'fecha_cruce_mensual', DatasetKeys.NUM_VT_BARRIO]]
                           .groupby(DatasetKeys.BARRIO, group_keys=True)
                           .apply(interpolate_group, include_groups=False)
                           .reset_index()
                           .rename(columns={'level_1': 'fecha_cruce_mensual'})) # Mantenemos el nombre coherente
        return df_interpolated
    
    @staticmethod
    def _porcentaje_vt(df_amaem_dom, df_interpolated):
        df_resampled = pd.merge(df_interpolated, df_amaem_dom, on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], how='left')
        df_resampled[DatasetKeys.PCT_VT_BARRIO] = ((df_resampled[DatasetKeys.NUM_VT_BARRIO] / df_resampled[DatasetKeys.NUM_CONTRATOS]) * 100)
        df_resampled[DatasetKeys.PCT_VT_BARRIO] = df_resampled[DatasetKeys.PCT_VT_BARRIO].fillna(0).round(2)
        df_resampled[DatasetKeys.NUM_VT_BARRIO] = df_resampled[DatasetKeys.NUM_VT_BARRIO].round().astype(int)

        return df_resampled[[DatasetKeys.BARRIO, 'fecha_cruce_mensual', DatasetKeys.NUM_VT_BARRIO, DatasetKeys.PCT_VT_BARRIO]]

    @staticmethod
    def _process_municipios(df_amaem: pd.DataFrame) -> pd.DataFrame:
        df_barrio               = INETourismProcessor._map_mun2barrios()
        df_amaem_dom, df_merge  = INETourismProcessor._merge_domesticos_ine(df_amaem, df_barrio)
        df_interpolated         = INETourismProcessor._interpolacion_mensual(df_merge)
        df_final_mun            = INETourismProcessor._porcentaje_vt(df_amaem_dom, df_interpolated) 

        return df_final_mun


    ######################################################################################
    #                                        PROVINCIA
    @staticmethod
    def _process_provincia() -> pd.DataFrame:
        df_prov = pd.read_csv(Paths.INE_PROVINCIA_VT, encoding="utf-8", sep=";")
        
        df_prov.columns = (df_prov.columns
                           .str.strip().str.lower()
                           .str.replace(' ', '_').str.replace(':', '')
                           .str.replace('á', 'a').str.replace('é', 'e')
                           .str.replace('í', 'i').str.replace('ó', 'o')
                           .str.replace('ú', 'u'))
        
        df_prov = df_prov.rename(columns={
            "fecha": "fecha_orig",
            "total_numero_de_alojamientos_turisticos_ocupados": DatasetKeys.OCUPACIONES_VT_PROV,
            "numero_de_noches_ocupadas": DatasetKeys.PERNOCTACIONES_VT_PROV
        })

        # Creamos 'fecha_cruce_mensual'
        df_prov['fecha_cruce_mensual'] = pd.to_datetime(df_prov['fecha_orig'].str.replace('M', '-')).dt.to_period('M')
        df_prov_clean = df_prov[['fecha_cruce_mensual', DatasetKeys.OCUPACIONES_VT_PROV, DatasetKeys.PERNOCTACIONES_VT_PROV]].copy()

        for col in [DatasetKeys.OCUPACIONES_VT_PROV, DatasetKeys.PERNOCTACIONES_VT_PROV]:
            if df_prov_clean[col].dtype == 'object':
                df_prov_clean[col] = df_prov_clean[col].str.replace('.', '', regex=False).astype(float)
            elif df_prov_clean[col].dtype in ['float64', 'int64']:
                df_prov_clean[col] = (df_prov_clean[col] * 1000).astype(int)

        return df_prov_clean