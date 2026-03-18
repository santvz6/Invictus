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
    

    ### --- Estructura de Datos Base --- ###
    DATA_DIR = INTERNAL / "data"
    PROC_DIR = DATA_DIR / "processed"
    RAW_DIR = DATA_DIR / "raw"
    EXT_DIR = DATA_DIR / "external"
    CONFIG_DIR = DATA_DIR / "config"

    ## --- CONFIG ---
    MAPPING_BARRIOS = CONFIG_DIR / "mapping_barrios.csv"
    MAPPING_BARRIOS_YAML = CONFIG_DIR / "mapping_barrios.yaml"
    
    ## --- RAW --- ##
    RAW_CSV_DIR = RAW_DIR / "csv"
    RAW_JSON_DIR = RAW_DIR / "json"
    # --- CSV --- #
    RAW_CSV_AMAEM = RAW_CSV_DIR / "AMAEM-2022-2024.csv"
    RAW_CSV_TELELECTURA = RAW_CSV_DIR / "contadores-telelectura-instalados.csv"
    # ... (el resto de los CSVs) ...
    # ----  JSON --- #
    RAW_JSON_BOCAS_HIDRANTES         = RAW_JSON_DIR / "bocas-de-hidrantes.json"
    RAW_JSON_CENTROS_BOMBEO          = RAW_JSON_DIR / "centros-de-bombeo.json"
    RAW_JSON_DEPOSITOS               = RAW_JSON_DIR / "depositos.json"
    RAW_JSON_ENTIDADES_POBLACION     = RAW_JSON_DIR / "entidades-de-poblacion.json"
    RAW_JSON_FUENTES                 = RAW_JSON_DIR / "fuentes.json"
    RAW_JSON_GRANDES_COLECTORES      = RAW_JSON_DIR / "grandes-colectores.json"
    RAW_JSON_IMBORNALES_GRAN         = RAW_JSON_DIR / "imbornales-de-gran-capacidad.json"
    RAW_JSON_IMBORNALES              = RAW_JSON_DIR / "imbornales.json"
    RAW_JSON_PLUVIOMETROS            = RAW_JSON_DIR / "pluviometros.json"
    RAW_JSON_REDES_ARTERIALES        = RAW_JSON_DIR / "redes-arteriales.json"
    RAW_JSON_REDES_PRIMARIAS         = RAW_JSON_DIR / "redes-primarias.json"
    RAW_JSON_SECTORES_CONSUMO        = RAW_JSON_DIR / "sectores-de-consumo.json"
    RAW_JSON_TUBERIAS_REGENERADA     = RAW_JSON_DIR / "tuberias-agua-regenerada.json"
    RAW_JSON_TUBERIAS_ALCANTARILLADO = RAW_JSON_DIR / "tuberias-de-alcantarillado-y-pluviales.json"
    RAW_JSON_ZONAS_VERDES            = RAW_JSON_DIR / "zonas-verdes.json"

    ## --- PROCESSED --- ##
    PROC_CSV_DIR = PROC_DIR / "csv"
    PROC_JSON_DIR = PROC_DIR / "json"
    # --- CSV --- #
    PROC_CSV_AMAEM = PROC_CSV_DIR / "AMAEM-2022-2024.csv"

    ## --- EXTERNAL --- ##

    # --- GVA --- #
    GVA_DIR = EXT_DIR / "gva"
    GVA_VIVIENDAS = GVA_DIR / "m-viviendas-2022-2025.csv"
    GVA_HOTELES   = GVA_DIR / "m-hoteles-2022-2026.csv"

    # --- INE --- #
    INE_DIR = EXT_DIR / "ine"

    INE_COMUNIDAD_TIPO_ALOJ = INE_DIR / "comunidad-info-tipo-aloj.csv"
    INE_COMUNIDAD_TOTAL     = INE_DIR / "comunidad-info-total.csv"
    INE_MUNICIPIOS_PLAZAS   = INE_DIR / "municipios-plazas-vt.csv"
    INE_MUNICIPIOS_PORCENT  = INE_DIR / "municipios-porcentaje-vt.csv"
    INE_PROVINCIA_HOTEL     = INE_DIR / "provincia-info-hotel.csv"
    INE_PROVINCIA_VT        = INE_DIR / "provincia-info-vt.csv"


    @classmethod
    def init_project(cls):
        dirs = [
            cls.INTERNAL,
            cls.DATA_DIR,  cls.EXPERIMENTS_DIR, cls.LOGS_DIR, cls.TEMP_DIR, 
            cls.CONFIG_DIR,
            cls.PROC_DIR, cls.PROC_CSV_DIR, cls.PROC_JSON_DIR
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


