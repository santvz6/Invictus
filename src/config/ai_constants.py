class AIConstants:
    """
    Parámetros técnicos y umbrales para los modelos de Machine Learning.
    Centralizamos aquí los valores que afectan al entrenamiento y la lógica de detección de anomalías.
    """
    
    # Semilla para garantizar la reproducibilidad de los experimentos y modelos
    RANDOM_STATE = 80
    
    # Número de grupos predeterminado para el clustering de barrios/sectores
    N_CLUSTERS_DEFAULT = 2

    # Modelo de lenguaje predeterminado para reportes analíticos locales (Ollama)
    # Cambiado de llama3.2 a qwen3 para mejor razonamiento y cumplimiento de licencias.
    LLM_MODEL = "qwen3"

    # Percentiles para definir el corte de anomalías (Autoencoder y Riesgo de Fraude)
    # Valores por encima de este umbral se consideran sospechosos.
    AE_ANOMALIES_PERCENTILE = 90
    FRAUD_RISK_PERCENTILE   = 90