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

    ES_FINDE = "es_finde"
    MES = "mes"
    
    # --- Columnas de salida de modelos ---
    CLUSTER = "cluster"
    ETIQUETA_IA = "etiqueta_ia"
    STATUS = "detect_status"
    CONFIDENCE = "confidence_score"


    @classmethod
    def get_feature_columns(cls):
        return [f"H{i}" for i in range(24)] + [cls.RATIO_WEEKEND, cls.MEAN_CONSUMO, cls.STD_CONSUMO]


class BusinessLabels:
    """Etiquetas asignadas por el Labeler (Fase 2)"""
    TURISTICO = "Turístico"
    DESOCUPADO = "Desocupado"
    INDUSTRIAL_FUGA = "Industrial/Fuga"
    RESIDENCIAL = "Residencial"

class FraudStatus:
    """Estados resultantes del FraudDetector (Fase 3)"""
    SOSPECHA_TURISTICO = "SOSPECHA_TURISTICO"
    ALERTA_TECNICA = "ALERTA_TECNICA"
    CONTRATO_DOMESTICO = "Doméstico"
    OK = "OK"


