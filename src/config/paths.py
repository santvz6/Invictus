import os
import shutil

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent # ../src/config/paths.py (3 levels)


class Paths:
    ROOT = BASE_DIR
    SRC = ROOT / "src"
    INTERNAL = ROOT / "internal"
   
    EXPERIMENTS_DIR = INTERNAL / "experiments"
    LOGS_DIR = INTERNAL / "logs"
    TEMP_DIR = INTERNAL / "temp"
    PROCESSED_DIR = INTERNAL / "processed"
    DATA_DIR = INTERNAL / "data"
    
    # CSV
    CSV_DIR = DATA_DIR / "csv"
    CSV_AMAEM = CSV_DIR / "AMAEM-2022-2024.csv"
    CSV_TELELECTURA = CSV_DIR / "contadores-telelectura-instalados.csv"

    # JSON
    JSON_DIR                        = DATA_DIR / "json"
    JSON_BOCAS_HIDRANTES            = JSON_DIR / "bocas-de-hidrantes.json"
    JSON_CENTROS_BOMBEO             = JSON_DIR / "centros-de-bombeo.json"
    JSON_DEPOSITOS                  = JSON_DIR / "depositos.json"
    JSON_ENTIDADES_POBLACION        = JSON_DIR / "entidades-de-poblacion.json"
    JSON_FUENTES                    = JSON_DIR / "fuentes.json"
    JSON_GRANDES_COLECTORES         = JSON_DIR / "grandes-colectores.json"
    JSON_IMBORNALES_GRAN            = JSON_DIR / "imbornales-de-gran-capacidad.json"
    JSON_IMBORNALES                 = JSON_DIR / "imbornales.json"
    JSON_PLUVIOMETROS               = JSON_DIR / "pluviometros.json"
    JSON_REDES_ARTERIALES           = JSON_DIR / "redes-arteriales.json"
    JSON_REDES_PRIMARIAS            = JSON_DIR / "redes-primarias.json"
    JSON_SECTORES_CONSUMO           = JSON_DIR / "sectores-de-consumo.json"
    JSON_TUBERIAS_REGENERADA        = JSON_DIR / "tuberias-agua-regenerada.json"
    JSON_TUBERIAS_ALCANTARILLADO    = JSON_DIR / "tuberias-de-alcantarillado-y-pluviales.json"
    JSON_ZONAS_VERDES               = JSON_DIR / "zonas-verdes.json"


    @classmethod
    def init_project(cls):
        dirs = [
            cls.INTERNAL,
            cls.DATA_DIR,  cls.EXPERIMENTS_DIR, cls.LOGS_DIR, cls.TEMP_DIR, cls.PROCESSED_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        cls._rotate_logs()

    @classmethod
    def _rotate_logs(cls, max_logs=10):
        existing_logs = sorted(
            [f for f in cls.LOGS_DIR.glob("*.log")], 
            key=os.path.getmtime
        )
        if len(existing_logs) >= max_logs:
            for log_file in existing_logs[:(len(existing_logs) - max_logs + 1)]:
                shutil.move(str(log_file), str(cls.TEMP_DIR / log_file.name))


