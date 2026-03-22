class DatasetKeys:
    # --- Columnas originales ---
    BARRIO          = "barrio"
    FECHA           = "fecha"
    CONSUMO         = "consumo"
    NUM_CONTRATOS   = "num_contratos"
    USO             = "uso"

    # --- OneHot Encoding ---    
    #USO_COMERCIAL = USO + "_COMERCIAL"
    #USO_DOMESTICO = USO + "_DOMESTICO"
    #USO_NO_DOMESTICO = USO + "_NO DOMESTICO"

    # --- Columnas calculadas (Features) ---
    CONSUMO_RATIO = "consumo_ratio"
    
    MES = "mes"
    MES_SIN = "mes_sin"
    MES_COS = "mes_cos"

    # --- Columnas Externas (Físicos) ---
    CONSUMO_FISICO_ESPERADO = "consumo_teorico_fisica"
    PREDICCION_FOURIER      = "prediccion_fourier"
    IMPACTO_EXOGENO         = "impacto_exogeno"
    RESIDUO                 = "residuo"
    NDVI_SATELITE           = "ndvi_satelite"
    
    # --- Columnas Externas (INE Turismo) ---
    NUM_VT_BARRIO_INE       = "num_vt_barrio"
    PCT_VT_BARRIO_INE       = "porcentaje_vt_barrio %"
    OCUP_VT_PROV_INE        = "ocupaciones_vt_prov"
    PERNOCT_VT_PROV_INE     = "pernoctaciones_vt_prov"

    # --- Columnas Externas (AEMET) ---
    TEMP_MEDIA              = "temperatura_media"
    PRECIPITACION           = "precipitacion"

    # --- Columnas Externas (GVA) ---
    NUM_VT_BARRIO_GVA       = "num_viviendas_barrio_gva"
    NUM_HOTELES_BARRIO_GVA  = "num_hoteles_barrio_gva"
    NUM_VT_ILEGALES         = "num_vt_ilegales"
    PCT_VT_ILEGALES         = "porcentaje_vt_ilegales %"

    PLAZAS_VIVIENDAS_GVA      = "plazas_viviendas_barrio_gva"
    PLAZAS_HOTELES_BARRIO_GVA = "plazas_hoteles_barrio_gva"

    # --- Columnas de Resultados (Modelos y Pipeline) ---
    CLUSTER                 = "cluster"
    RECONSTRUCTION_ERROR    = "reconstruction_error"
    AE_SCORE                = "ae_score"
    PHYSICS_SCORE           = "physics_score"
    RESIDUO_POSITIVO        = "residuo_positivo"
    FRAUD_RISK_SCORE        = "FRAUD_RISK_SCORE"
    ALERTA_TURISTICA_ILEGAL = "ALERTA_TURISTICA_ILEGAL"
    IS_AE_ANOMALY           = "is_ae_anomaly"
    IS_PHYSICS_ANOMALY      = "is_physics_anomaly"
    
    # --- Columnas de Reporte ---
    NIVEL_RIESGO            = "nivel_riesgo"
    MOTIVO                  = "motivo"