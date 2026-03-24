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
    
    # --- Datos de Turismo y Vivienda (INE) ---
    NUM_VT_BARRIO_INE       = "num_vt_barrio"
    PCT_VT_BARRIO_INE       = "porcentaje_vt_barrio %"
    OCUP_VT_PROV_INE        = "ocupaciones_vt_prov"
    PERNOCT_VT_PROV_INE     = "pernoctaciones_vt_prov"


    # --- Factores Meteorológicos (AEMET) ---
    TEMP_MEDIA              = "temperatura_media"
    PRECIPITACION           = "precipitacion"

    # --- Datos de Turismo y Vivienda (GVA) ---
    NUM_VT_BARRIO_GVA       = "num_viviendas_barrio_gva"
    NUM_HOTELES_BARRIO_GVA  = "num_hoteles_barrio_gva"
    NUM_VT_SIN_REGISTRAR    = "num_vt_sin_registrar"
    PCT_VT_SIN_REGISTRAR    = "porcentaje_vt_sin registrar %"

    PLAZAS_VIVIENDAS_GVA      = "plazas_viviendas_barrio_gva"
    PLAZAS_HOTELES_BARRIO_GVA = "plazas_hoteles_barrio_gva"

    # --- Identificadores resultantes de Modelos de ML y Pipelines ---
    CLUSTER                 = "cluster"
    RECONSTRUCTION_ERROR    = "reconstruction_error"
    
    # Métricas de detección de anomalías
    RESIDUO_POSITIVO        = "residuo_positivo"
    FRAUD_RISK_SCORE        = "FRAUD_RISK_SCORE"
    ALERTA_TURISTICA_ILEGAL = "ALERTA_TURISTICA_ILEGAL"
    
    # Flags Booleanos de Anomalía
    IS_PHYSICS_ANOMALY      = "is_physics_anomaly"
    IS_GENERAL_ANOMALY      = "is_general_anomaly"
    IS_WEIGHTED_ANOMALY     = "is_weighted_anomaly"
    AE_SCORE_GENERAL        = "ae_score_general"
    AE_SCORE_WEIGHTED       = "ae_score_weighted"
    PHYSICS_SCORE           = "physics_score"   # Escala 0-100 (100 = Umbral de Alerta)
    
    # --- Campos para generación de Reportes Finales ---
    NIVEL_RIESGO            = "nivel_riesgo"
    MOTIVO                  = "motivo"