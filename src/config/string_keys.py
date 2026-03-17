class DatasetKeys:
    # --- Columnas originales ---
    BARRIO          = "barrio"
    FECHA           = "fecha"
    CONSUMO         = "consumo"
    NUM_CONTRATOS   = "num_contratos"
    USO             = "uso"

    # --- OneHot Encoding ---    
    USO_COMERCIAL = USO + "_COMERCIAL"
    USO_DOMESTICO = USO + "_DOMESTICO"
    USO_NO_DOMESTICO = USO + "_NO DOMESTICO"

    # --- Columnas calculadas (Features) ---
    CONTRATO_RATIO = "consumo_ratio"
    MES = "mes"
    CONSUMO_FISICO_ESPERADO = "consumo_teorico_fisica"
    
    # --- Columnas Externas (INE Turismo) ---
    NUM_VT_BARRIO      = "num_vt_barrio"
    PCT_TURISTICO_REAL = "porcentaje_turistico_real"
    OCUPACIONES_VT     = "ocupaciones_vt"
    PERNOCTACIONES_VT  = "pernoctaciones_vt"