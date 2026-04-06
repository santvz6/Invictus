"""
Paquete de configuración centralizada para el proyecto Invictus.
Expone las clases y funciones principales para la gestión de rutas, constantes de IA,
mapeo de datos y sistema de logs.
"""

from .ai_constants import AIConstants
from .logging import get_logger
from .paths import Paths
from .string_keys import DatasetKeys
from .features import FeatureConfig, FeatureScaling
from .ollama_client import OllamaLLM