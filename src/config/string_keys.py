class DatasetKeys:
    """
    Diccionario centralizado de nombres de columnas (keys) utilizadas en los DataFrames.
    Evita el uso de strings literales dispersos por el código, facilitando cambios en el esquema.
    """
    
    # --- Columnas procedentes de los datos originales (AMAEM) ---
    BARRIO          = "barrio"
    FECHA           = "fecha"
    CONSUMO         = "consumo"
    NUM_CONTRATOS   = "num_contratos"
    USO             = "uso"

    # --- Variables de Ingeniería de Características (Features) ---
    CONSUMO_RATIO = "consumo_ratio"
    
    # Componentes temporales cíclicas
    MES = "mes"
    MES_SIN = "mes_sin"
    MES_COS = "mes_cos"

    # --- Modelado Físico y Análisis de Residuos ---
    CONSUMO_FISICO_ESPERADO = "consumo_teorico_fisica"
    PREDICCION_FOURIER      = "prediccion_fourier"
    IMPACTO_EXOGENO         = "impacto_exogeno"
    RESIDUO                 = "residuo"
    NDVI_SATELITE           = "ndvi_satelite"
    
    # --- Factores Meteorológicos (AEMET) ---
    TEMP_MEDIA              = "temperatura_media"
    PRECIPITACION           = "precipitacion"

    # --- Datos de Turismo y Vivienda (INE) ---
    NUM_VT_BARRIO_INE       = "num_vt_barrio"
    PCT_VT_BARRIO_INE       = "porcentaje_vt_barrio %"
    OCUP_VT_PROV_INE        = "ocupaciones_vt_prov"
    PERNOCT_VT_PROV_INE     = "pernoctaciones_vt_prov"
    

    # --- Datos de Turismo y Vivienda (GVA) ---
    NUM_VT_BARRIO_GVA       = "num_viviendas_barrio_gva"
    NUM_HOTELES_BARRIO_GVA  = "num_hoteles_barrio_gva"
    NUM_VT_SIN_REGISTRAR    = "num_vt_sin_registrar"
    PCT_VT_SIN_REGISTRAR    = "porcentaje_vt_sin registrar %"

    PLAZAS_VIVIENDAS_GVA      = "plazas_viviendas_barrio_gva"
    PLAZAS_HOTELES_BARRIO_GVA = "plazas_hoteles_barrio_gva"

    # --- Datos de Festivos ---
    DIAS_FESTIVOS = "dias_festivos"
    PCT_FESTIVOS  = "pct_festivos"
    
    # --- Causas de Anomalías (Físicos 6 Niveles) ---
    Z_ERROR_FINAL = "z_error_final"
    ALERTA_NIVEL = "alerta_nivel"
    
    PCT_CALOR_FRIO = "pct_calor_frio"
    PCT_LLUVIA_SEQUIA = "pct_lluvia_sequia"
    PCT_VEGETACION = "pct_vegetacion"
    PCT_TURISMO = "pct_turismo"
    PCT_FIESTA = "pct_fiesta"
    PCT_CAUSA_DESCONOCIDA = "pct_causa_desconocida"