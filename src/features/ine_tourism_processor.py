"""
Módulo de procesamiento de métricas turísticas del Instituto Nacional de Estadística (INE).

Este componente integra datos externos sobre Viviendas Turísticas (VT), ocupación y 
pernoctaciones, permitiendo modelar la presión turística tanto a nivel municipal 
(interpolada por barrio) como provincial.
"""

import pandas as pd
from src.config import DatasetKeys, Paths, get_logger
logger = get_logger(__name__)

class INETourismProcessor:
    """
    Procesador de microdatos y series temporales del INE.
    
    Gestiona la complejidad de las diferentes escalas geográficas (municipio vs provincia)
    y temporales (datos trimestrales o puntuales) mediante técnicas de interpolación 
    lineal y pesaje ponderado por barrio.
    """

    @staticmethod
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """
        Orquesta el flujo completo de enriquecimiento con datos del INE.

        Args:
            df_amaem (pd.DataFrame): Dataset principal de AMAEM.

        Returns:
            pd.DataFrame: Dataset enriquecido con métricas de VT, ocupación y pernoctaciones:
                - NUM_VT_BARRIO_INE
                - PCT_VT_BARRIO_INE
                - OCUP_VT_PROV_INE
                - PERNOCT_VT_PROV_INE
        """
        logger.info("Iniciando enriquecimiento con datos turísticos del INE...")
        
        # Generación de copia y anclaje temporal mensual para garantizar la integridad del cruce
        df_final = df_amaem.copy()
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')

        # 1. Pipeline Municipal: Distribución de VT de municipios hacia barrios
        # Se requiere df_final para normalizar según el número de contratos domésticos
        df_municipios = INETourismProcessor._process_municipios(df_final)
        
        # 2. Pipeline Provincial: Métricas generales de ocupación y pernoctaciones
        df_provincia = INETourismProcessor._process_provincia()

        # 3. Integración Geográfica Dual (Merge)
        logger.info("Cruzando datos de AMAEM con INE Municipios y Provincia...")
        
        # Cruce Municipal: Basado en Barrio y Periodo Mensual
        df_final = pd.merge(
            df_final, 
            df_municipios, 
            on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], 
            how='left'
        )
        
        # Cruce Provincial: Basado únicamente en el Periodo Mensual (clima macro)
        df_final = pd.merge(
            df_final, 
            df_provincia, 
            on='fecha_cruce_mensual', 
            how='left'
        )

        # Imputación de nulos residuales tras el cruce (se asume 0 para periodos sin cobertura INE)
        cols_ine = [
            DatasetKeys.NUM_VT_BARRIO_INE, 
            DatasetKeys.PCT_VT_BARRIO_INE, 
            DatasetKeys.OCUP_VT_PROV_INE, 
            DatasetKeys.PERNOCT_VT_PROV_INE
        ]
        for col in cols_ine:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        # Limpieza de variables técnicas de procesamiento
        df_final = df_final.drop(columns=['fecha_cruce_mensual'])
        
        # Registro del punto de control intermedio
        logger.info(f"Registrando dataset intermedio INE en {Paths.PROC_CSV_STEP_INE}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + [c for c in cols_ine if c in df_final.columns]
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP_INE, index=False)
        
        logger.info("Enriquecimiento con INE completado.")
        return df_final

    ######################################################################################
    #                         GESTIÓN DE DATOS MUNICIPALES (VT)
    ######################################################################################

    @staticmethod
    def _map_mun2barrios() -> pd.DataFrame:
        """
        Realiza el 'weighting' o distribución pesada de datos municipales a nivel de barrio.

        Utiliza un mapeo predefinido para repartir el total de viviendas turísticas
        municipales basándose en pesos específicos por barrio.

        Returns:
            pd.DataFrame: Series temporales de VT por barrio y mes.
        """
        df_mun = pd.read_csv(Paths.INE_MUNICIPIOS_PLAZAS, encoding="latin1", sep="\t")
        
        # Garantizamos la disponibilidad del mapeo de barrios
        if not Paths.MAPPING_BARRIOS.exists():
            from src.config.barrio_mapping import export_yaml_to_csv
            export_yaml_to_csv()
            
        df_mapping = pd.read_csv(Paths.MAPPING_BARRIOS, sep=";")
        
        # Estandarización temporal
        df_mun['fecha_cruce_mensual'] = pd.to_datetime(df_mun['Periodo'].str.replace('M', '-')).dt.to_period('M')
        
        # Limpieza de volumenes numéricos (manejo de separadores de miles y decimales)
        if 'Total' in df_mun.columns:
            df_mun['Total'] = df_mun['Total'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_mun['Total'] = pd.to_numeric(df_mun['Total'], errors='coerce').fillna(0)

        # Filtrado específico para viviendas (excluyendo plazas)
        df_mun = df_mun[df_mun['Viviendas y plazas'].str.contains('Viviendas', na=False, case=False)]
        df_mun_data = df_mun[['Municipios', 'fecha_cruce_mensual', 'Total']].rename(columns={'Total': 'Total_vt_municipio'})

        # Cálculo de la cuota por barrio basada en el peso asignado en el mapping
        df_weighted = pd.merge(df_mapping, df_mun_data, left_on='municipio', right_on='Municipios')
        df_weighted['peso'] = pd.to_numeric(df_weighted['peso'], errors='coerce').fillna(0)
        df_weighted['Total_vt_municipio'] = pd.to_numeric(df_weighted['Total_vt_municipio'], errors='coerce').fillna(0)
        df_weighted[DatasetKeys.NUM_VT_BARRIO_INE] = df_weighted['Total_vt_municipio'] * df_weighted['peso']

        # Agregación por barrio y periodo
        df_barrio = df_weighted.groupby([DatasetKeys.BARRIO, 'fecha_cruce_mensual'])[DatasetKeys.NUM_VT_BARRIO_INE].sum().reset_index()
        return df_barrio
    
    @staticmethod
    def _merge_domesticos_ine(df_amaem: pd.DataFrame, df_barrio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aisla contratos domésticos y cruza con la penetración turística del barrio.

        Args:
            df_amaem (pd.DataFrame): Dataset base completo.
            df_barrio (pd.DataFrame): Datos de VT pesados por barrio.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (DF AMAEM Domésticos, DF Cruzado).
        """
        df_amaem_dom = df_amaem[df_amaem[DatasetKeys.USO] == 'DOMESTICO'][[DatasetKeys.BARRIO, 'fecha_cruce_mensual', DatasetKeys.NUM_CONTRATOS]]
        df_merge = pd.merge(df_barrio, df_amaem_dom, on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], how='left')
        return df_amaem_dom, df_merge

    @staticmethod
    def _interpolacion_mensual(df_merge: pd.DataFrame) -> pd.DataFrame:
        """
        Completa los vacíos temporales mediante interpolación lineal de series históricas.
        
        Dado que el INE puede no reportar datos mensualmente, esta función genera 
        continuidad en la serie histórica para evitar saltos artificiales en el modelo.

        Args:
            df_merge (pd.DataFrame): Dataset crudo tras el merge.

        Returns:
            pd.DataFrame: Dataset interpolado linealmente a nivel mensual.
        """
        periodo_minimo = df_merge['fecha_cruce_mensual'].min()
        rango_completo = pd.period_range(start=periodo_minimo, end='2024-12', freq='M')

        def interpolate_group(group):
            # Reindexamos para forzar la aparición de todos los meses y aplicamos interpolación
            group = group.set_index('fecha_cruce_mensual').reindex(rango_completo)
            if DatasetKeys.NUM_VT_BARRIO_INE in group.columns:
                group[DatasetKeys.NUM_VT_BARRIO_INE] = group[DatasetKeys.NUM_VT_BARRIO_INE].interpolate(method='linear')
            # Retornamos el periodo de interés para el modelo
            return group.loc['2022-01':'2024-12']

        df_interpolated = (df_merge[[DatasetKeys.BARRIO, 'fecha_cruce_mensual', DatasetKeys.NUM_VT_BARRIO_INE]]
                           .groupby(DatasetKeys.BARRIO, group_keys=True)
                           .apply(interpolate_group, include_groups=False)
                           .reset_index()
                           .rename(columns={'level_1': 'fecha_cruce_mensual'}))
        return df_interpolated
    
    @staticmethod
    def _porcentaje_vt(df_amaem_dom: pd.DataFrame, df_interpolated: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el porcentaje relativo de VT respecto al número total de contratos.

        Args:
            df_amaem_dom (pd.DataFrame): Datos de contratación doméstica.
            df_interpolated (pd.DataFrame): Datos de VT interpolados.

        Returns:
            pd.DataFrame: Dataset final municipal con métricas absolutas y porcentuales.
        """
        df_resampled = pd.merge(df_interpolated, df_amaem_dom, on=[DatasetKeys.BARRIO, 'fecha_cruce_mensual'], how='left')
        
        # Ingeniería de características: Ratio de Vivienda Turística / Contratos Totales
        df_resampled[DatasetKeys.PCT_VT_BARRIO_INE] = ((df_resampled[DatasetKeys.NUM_VT_BARRIO_INE] / df_resampled[DatasetKeys.NUM_CONTRATOS]) * 100)
        df_resampled[DatasetKeys.PCT_VT_BARRIO_INE] = df_resampled[DatasetKeys.PCT_VT_BARRIO_INE].fillna(0).round(2)
        df_resampled[DatasetKeys.NUM_VT_BARRIO_INE] = df_resampled[DatasetKeys.NUM_VT_BARRIO_INE].round().astype(int)

        return df_resampled[[DatasetKeys.BARRIO, 'fecha_cruce_mensual', DatasetKeys.NUM_VT_BARRIO_INE, DatasetKeys.PCT_VT_BARRIO_INE]]

    @staticmethod
    def _process_municipios(df_amaem: pd.DataFrame) -> pd.DataFrame:
        """
        Encapsula el flujo completo de procesamiento de datos por municipio.

        Args:
            df_amaem (pd.DataFrame): Dataset original de AMAEM.

        Returns:
            pd.DataFrame: Datos municipales listos para el merge final.
        """
        df_barrio               = INETourismProcessor._map_mun2barrios()
        df_amaem_dom, df_merge  = INETourismProcessor._merge_domesticos_ine(df_amaem, df_barrio)
        df_interpolated         = INETourismProcessor._interpolacion_mensual(df_merge)
        df_final_mun            = INETourismProcessor._porcentaje_vt(df_amaem_dom, df_interpolated) 

        return df_final_mun

    ######################################################################################
    #                         GESTIÓN DE DATOS PROVINCIALES
    ######################################################################################

    @staticmethod
    def _process_provincia() -> pd.DataFrame:
        """
        Limpia y tipifica las series temporales de ocupación hotelera a nivel provincial.

        Returns:
            pd.DataFrame: Series mensuales de pernoctaciones y ocupación provincial.
        """
        df_prov = pd.read_csv(Paths.INE_PROVINCIA_VT, encoding="utf-8", sep=";")
        
        # Normalización agresiva de cabeceras para eliminar acentos y caracteres especiales
        df_prov.columns = (df_prov.columns
                           .str.strip().str.lower()
                           .str.replace(' ', '_').str.replace(':', '')
                           .str.replace('á', 'a').str.replace('é', 'e')
                           .str.replace('í', 'i').str.replace('ó', 'o')
                           .str.replace('ú', 'u'))
        
        # Mapeo a las claves estándar del sistema
        df_prov = df_prov.rename(columns={
            "fecha": "fecha_orig",
            "total_numero_de_alojamientos_turisticos_ocupados": DatasetKeys.OCUP_VT_PROV_INE,
            "numero_de_noches_ocupadas": DatasetKeys.PERNOCT_VT_PROV_INE
        })

        # Sincronización temporal
        df_prov['fecha_cruce_mensual'] = pd.to_datetime(df_prov['fecha_orig'].str.replace('M', '-')).dt.to_period('M')
        df_prov_clean = df_prov[['fecha_cruce_mensual', DatasetKeys.OCUP_VT_PROV_INE, DatasetKeys.PERNOCT_VT_PROV_INE]].copy()

        # Corrección de formatos numéricos (manejo de escalas de miles si vienen pre-formateadas)
        for col in [DatasetKeys.OCUP_VT_PROV_INE, DatasetKeys.PERNOCT_VT_PROV_INE]:
            if df_prov_clean[col].dtype == 'object':
                df_prov_clean[col] = df_prov_clean[col].str.replace('.', '', regex=False).astype(float)
            elif df_prov_clean[col].dtype in ['float64', 'int64']:
                # Ajuste potencial de escala si los datos son porcentajes de miles (según formato INE)
                df_prov_clean[col] = (df_prov_clean[col] * 1000).astype(int)

        return df_prov_clean