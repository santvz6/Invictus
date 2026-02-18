import logging
from datetime import datetime
from src.config.paths import Paths

########################## LOGGER SETUP ##########################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_LOG_FILE = Paths.LOGS_DIR / f"invictus_{timestamp}.log"

def get_logger(module_name):
    logger = logging.getLogger(module_name)


    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Formato único para consola y archivo
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Manejador de Consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Manejador de Archivo (si existe el directorio)
        if Paths.LOGS_DIR.exists():
            file_handler = logging.FileHandler(CURRENT_LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger