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
    MES_SIN = "mes_sin"
    MES_COS = "mes_cos"

    # --- Columnas Externas (Físicos) ---
    CONSUMO_FISICO_ESPERADO = "consumo_teorico_fisica"
    
    # --- Columnas Externas (INE Turismo) ---
    NUM_VT_BARRIO           = "num_vt_barrio"
    PCT_VT_BARRIO           = "porcentaje_vt_barrio"
    OCUPACIONES_VT_PROV     = "ocupaciones_vt_prov"
    PERNOCTACIONES_VT_PROV  = "pernoctaciones_vt_prov"