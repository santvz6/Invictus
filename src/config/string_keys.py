class DataSchema:
    # --- Columnas originales ---
    ID_CONTADOR = "ID_Contador"
    TIMESTAMP = "Timestamp"
    CONSUMO = "Consumo_m3"
    CONTRATO = "Tipo_Contrato" 

    # --- Columnas calculadas (Features) ---
    HORA = "Hora"
    ES_FINDE = "es_finde"
    RATIO_WEEKEND = "ratio_wk"
    MEAN_CONSUMO = "mean"
    STD_CONSUMO = "std"
    
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