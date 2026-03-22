import pandas as pd
import logging

from src.config import DatasetKeys, Paths

logger = logging.getLogger(__name__)

class GVAProcessor:
    """
    Clase encargada de procesar, limpiar y cruzar los datos externos de la 
    Generalitat Valenciana (GVA) referentes a Viviendas Turísticas y Hoteles
    calculando las métricas activas por mes evaluando 'Fecha Alta' y 'Fecha Baja'.
    """

    @staticmethod
    def process(df_amaem: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando enriquecimiento con datos turísticos de la GVA (Altas y Bajas)...")
        
        # 0. Crear copia y ancla mensual para el cruce
        df_final = df_amaem.copy()
        df_final[DatasetKeys.FECHA] = pd.to_datetime(df_final[DatasetKeys.FECHA])
        df_final['fecha_cruce_mensual'] = df_final[DatasetKeys.FECHA].dt.to_period('M')

        meses_unicos = df_final['fecha_cruce_mensual'].unique()

        # 1. Procesar Viviendas Turísticas (VT)
        df_vt = GVAProcessor._process_file(
            filepath=Paths.GVA_VIVIENDAS, 
            meses_unicos=meses_unicos, 
            prefix='viviendas'
        )

        # 2. Procesar Hoteles
        df_hoteles = GVAProcessor._process_file(
            filepath=Paths.GVA_HOTELES, 
            meses_unicos=meses_unicos, 
            prefix='hoteles'
        )

        # 3. Cruzar con el DataFrame principal
        if not df_vt.empty:
            df_vt = df_vt.rename(columns={
                'num_viviendas_gva': DatasetKeys.NUM_VT_BARRIO_GVA,
                'plazas_viviendas_gva': DatasetKeys.PLAZAS_VIVIENDAS_GVA
            })
            df_final = pd.merge(df_final, df_vt, on='fecha_cruce_mensual', how='left')
        else:
            df_final[DatasetKeys.NUM_VT_BARRIO_GVA] = 0
            df_final[DatasetKeys.PLAZAS_VIVIENDAS_GVA] = 0

        if not df_hoteles.empty:
            df_hoteles = df_hoteles.rename(columns={
                'num_hoteles_gva': DatasetKeys.NUM_HOTELES_BARRIO_GVA,
                'plazas_hoteles_gva': DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA
            })
            df_final = pd.merge(df_final, df_hoteles, on='fecha_cruce_mensual', how='left')
        else:
            df_final[DatasetKeys.NUM_HOTELES_BARRIO_GVA] = 0
            df_final[DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA] = 0

        # Limpiar nulos (por si hay meses que no cruzaron)
        cols_gva = [
            DatasetKeys.NUM_VT_BARRIO_GVA, 
            DatasetKeys.PLAZAS_VIVIENDAS_GVA, 
            DatasetKeys.NUM_HOTELES_BARRIO_GVA, 
            DatasetKeys.PLAZAS_HOTELES_BARRIO_GVA
        ]
        for col in cols_gva:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        # 4. Limpieza Final de columnas temporales
        df_final = df_final.drop(columns=['fecha_cruce_mensual'])

        logger.info(f"Guardando dataset intermedio en {Paths.PROC_CSV_STEP6_GVA}")
        cols_to_save = [DatasetKeys.BARRIO, DatasetKeys.FECHA] + cols_gva
        df_final[cols_to_save].drop_duplicates().to_csv(Paths.PROC_CSV_STEP6_GVA, index=False)

        return df_final

    @staticmethod
    def _process_file(filepath, meses_unicos, prefix):
        try:
            # Archivos separados por punto y coma, codificación que soporta tildes/ñ
            df = pd.read_csv(filepath, sep=';', encoding='latin1', on_bad_lines='skip')
        except FileNotFoundError:
            logger.error(f"No se encontró el archivo GVA en {filepath}. Se omitirá el cruce.")
            return pd.DataFrame()

        # Normalizar nombres de columnas (minúsculas, sin espacios)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        col_municipio = 'municipio'
        col_alta = 'fecha_alta'
        col_baja = 'fecha_baja'
        col_plazas = 'plazas'

        # Filtro 1: Conversión de fechas (formato dd/mm/yyyy)
        if col_alta in df.columns:
            df[col_alta] = pd.to_datetime(df[col_alta], format='%d/%m/%Y', dayfirst=True, errors='coerce')
        else:
            logger.warning(f"La columna {col_alta} no existe en {filepath}.")
            return pd.DataFrame()
            
        if col_baja in df.columns:
            df[col_baja] = pd.to_datetime(df[col_baja], format='%d/%m/%Y', dayfirst=True, errors='coerce')
        else:
            df[col_baja] = pd.NaT

        # Filtro 2: Limpieza numérica de Plazas
        if col_plazas in df.columns:
            df[col_plazas] = pd.to_numeric(df[col_plazas], errors='coerce').fillna(0)
        else:
            df[col_plazas] = 0

        # Filtro 3: Identificar número de apartamentos aportados por el registro
        # (Si es un bloque, la columna Nm. Apartamentos indica cuántas VT reales son)
        col_apt = next((c for c in df.columns if 'apartamentos' in c), None)
        if col_apt:
            df['peso_unidad'] = pd.to_numeric(df[col_apt], errors='coerce').fillna(1)
        else:
            df['peso_unidad'] = 1

        # Bucle: Contabilizar volumen activo por mes
        results = []
        for period in meses_unicos:
            month_start = period.to_timestamp(how='start')
            month_end = period.to_timestamp(how='end')

            # Condición de actividad:
            # - Alta registrada ANTES o DURANTE este mes
            # - NO tiene baja, o se dio de baja DESPUÉS del inicio de este mes
            cond_alta = df[col_alta].notna() & (df[col_alta] <= month_end)
            cond_baja = df[col_baja].isna() | (df[col_baja] >= month_start)

            active_df = df[cond_alta & cond_baja]
            
            results.append({
                'fecha_cruce_mensual': period,
                f'num_{prefix}_gva': active_df['peso_unidad'].sum(),
                f'plazas_{prefix}_gva': active_df[col_plazas].sum()
            })

        return pd.DataFrame(results)