"""
logging.py
----------
Sistema de registro (logging) centralizado del proyecto INVICTUS.
Configura la salida de trazas tanto por consola como por archivo persistente.
"""

import logging
from datetime import datetime
from src.config.paths import Paths

########################## CONFIGURACIÓN DEL LOGGER ##########################
# Generamos un timestamp para que cada ejecución tenga su propio archivo de log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_LOG_FILE = Paths.LOGS_DIR / f"invictus_{timestamp}.log"

def get_logger(module_name):
    """
    Configura y retorna una instancia de logger para el módulo especificado.
    
    Si el logger ya tiene manejadores configurados, se retorna tal cual para evitar
    mensajes duplicados. Configura salida tanto a consola como a archivo.
    """
    logger = logging.getLogger(module_name)

    if not logger.handlers:
        # Nivel predeterminado: DEBUG para capturar detalles durante el desarrollo
        logger.setLevel(logging.DEBUG)
        
        # Formato unificado: [Fecha Hora] - [NombreMódulo] - [Nivel] - Mensaje
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Configuración del manejador de consola (Standard Output)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Configuración del manejador de archivo (persistente)
        # Solo se activa si el directorio de logs existe
        if Paths.LOGS_DIR.exists():
            file_handler = logging.FileHandler(CURRENT_LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger