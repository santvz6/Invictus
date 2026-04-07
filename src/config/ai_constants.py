"""
ai_constants.py
---------------
Constantes técnicas y parámetros de hiper-parametrización para los modelos de IA.
Incluye la configuración para Ollama (LLM local) y semillas de aleatoriedad.
"""

class AIConstants:
    """
    Parámetros técnicos y umbrales para los modelos de Machine Learning.
    Centralizamos aquí los valores que afectan al entrenamiento y la lógica de detección de anomalías.
    """
    
    # Semilla para garantizar la reproducibilidad de los experimentos y modelos
    RANDOM_STATE = 80
    
    # ═════════════════════════════════════════════════════════════════════════
    # Configuración de Ollama (LLM Local - Qwen)
    # ═════════════════════════════════════════════════════════════════════════
    # Modelo de lenguaje predeterminado para reportes analíticos locales (Ollama)
    LLM_MODEL = "qwen:7b"  # Qwen 7B vía Ollama (~4.7GB)
    LLM_BASE_URL = "http://localhost:11434"  # URL del servidor Ollama
    LLM_TIMEOUT = 120  # Timeout en segundos para generación de texto
    LLM_TEMPERATURE = 0.7  # Creatividad: 0.0 (determinista) a 1.0 (creativo)
    LLM_TOP_K = 40  # Tokens a considerar en cada paso
    LLM_TOP_P = 0.9  # Nucleus sampling (0-1)

    # Percentiles para definir el corte de anomalías (Detección de Anomalías)
    FRAUD_RISK_PERCENTILE = 90