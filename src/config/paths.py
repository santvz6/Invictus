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
    PROC_DIR = INTERNAL / "processed"

    ### --- Estructura de Datos Base --- ###
    DATA_DIR = INTERNAL / "data"
    CONFIG_DIR = DATA_DIR / "config"

    ## --- CONFIG ---
    MAPPING_BARRIOS = CONFIG_DIR / "mapping_barrios.csv"
    MAPPING_BARRIOS_YAML = CONFIG_DIR / "mapping_barrios.yaml"
    
    ## --- AMAEM --- ##
    AMAEM_DIR = DATA_DIR / "amaem"
    AMAEM_CSV_DIR  = AMAEM_DIR / "csv"
    AMAEM_JSON_DIR = AMAEM_DIR / "json"
    # --- CSV --- #
    RAW_CSV_AMAEM = AMAEM_CSV_DIR / "AMAEM-2022-2024.csv"
    RAW_CSV_TELELECTURA = AMAEM_CSV_DIR / "contadores-telelectura-instalados.csv"
    # ... (el resto de los CSVs) ...
    # ----  JSON --- #
    RAW_JSON_BOCAS_HIDRANTES         = AMAEM_JSON_DIR / "bocas-de-hidrantes.json"
    RAW_JSON_CENTROS_BOMBEO          = AMAEM_JSON_DIR / "centros-de-bombeo.json"
    RAW_JSON_DEPOSITOS               = AMAEM_JSON_DIR / "depositos.json"
    RAW_JSON_ENTIDADES_POBLACION     = AMAEM_JSON_DIR / "entidades-de-poblacion.json"
    RAW_JSON_FUENTES                 = AMAEM_JSON_DIR / "fuentes.json"
    RAW_JSON_GRANDES_COLECTORES      = AMAEM_JSON_DIR / "grandes-colectores.json"
    RAW_JSON_IMBORNALES_GRAN         = AMAEM_JSON_DIR / "imbornales-de-gran-capacidad.json"
    RAW_JSON_IMBORNALES              = AMAEM_JSON_DIR / "imbornales.json"
    RAW_JSON_PLUVIOMETROS            = AMAEM_JSON_DIR / "pluviometros.json"
    RAW_JSON_REDES_ARTERIALES        = AMAEM_JSON_DIR / "redes-arteriales.json"
    RAW_JSON_REDES_PRIMARIAS         = AMAEM_JSON_DIR / "redes-primarias.json"
    RAW_JSON_SECTORES_CONSUMO        = AMAEM_JSON_DIR / "sectores-de-consumo.json"
    RAW_JSON_TUBERIAS_REGENERADA     = AMAEM_JSON_DIR / "tuberias-agua-regenerada.json"
    RAW_JSON_TUBERIAS_ALCANTARILLADO = AMAEM_JSON_DIR / "tuberias-de-alcantarillado-y-pluviales.json"
    RAW_JSON_ZONAS_VERDES            = AMAEM_JSON_DIR / "zonas-verdes.json"

    ## --- PROCESSED --- ##
    PROC_CSV_STEP_AMAEM     = PROC_DIR / "step_amaem.csv"
    PROC_CSV_STEP_INE       = PROC_DIR / "step_ine.csv"
    PROC_CSV_STEP_GVA       = PROC_DIR / "step_gva.csv"
    PROC_CSV_STEP_AEMET     = PROC_DIR / "step_aemet.csv"
    PROC_CSV_STEP_SENTINEL  = PROC_DIR / "step_sentinel.csv"

    PROC_CSV_AMAEM_FISICOS    = PROC_DIR / "AMAEM-2022-2024_fisicos.csv"
    PROC_CSV_AMAEM_SCALED     = PROC_DIR / "AMAEM-2022-2024_scaled.csv"
    PROC_CSV_AMAEM_NOT_SCALED = PROC_DIR / "AMAEM-2022-2024_not_scaled.csv"
    
    ## --- EXTERNAL --- ##

    # --- GVA --- #
    GVA_DIR = DATA_DIR / "gva"
    GVA_VIVIENDAS = GVA_DIR / "municipios-vt-2022-2024.csv"
    GVA_HOTELES   = GVA_DIR / "municipios-hoteles-2022-2024.csv"

    # --- INE --- #
    INE_DIR = DATA_DIR / "ine"

    INE_COMUNIDAD_TIPO_ALOJ = INE_DIR / "comunidad-info-tipo-aloj.csv"
    INE_COMUNIDAD_TOTAL     = INE_DIR / "comunidad-info-total.csv"
    INE_MUNICIPIOS_PLAZAS   = INE_DIR / "municipios-plazas-vt.csv"
    INE_MUNICIPIOS_PORCENT  = INE_DIR / "municipios-porcentaje-vt.csv"
    INE_PROVINCIA_HOTEL     = INE_DIR / "provincia-info-hotel.csv"
    INE_PROVINCIA_VT        = INE_DIR / "provincia-info-vt.csv"

    # --- AEMET ---
    AEMET_DIR               = DATA_DIR / "aemet"
    AEMET_CLIMA_BARRIOS     = AEMET_DIR / "clima_barrios_alicante_final.csv"

    # --- SENTINEL ---
    SENTINEL_DIR  = DATA_DIR / "sentinel"
    SENTINEL_NDVI = SENTINEL_DIR / "ndvi_alicante.csv"

    @classmethod
    def init_project(cls):
        dirs = [
            cls.INTERNAL,
            cls.DATA_DIR,  cls.EXPERIMENTS_DIR, cls.LOGS_DIR, cls.TEMP_DIR, 
            cls.CONFIG_DIR,
            cls.PROC_DIR, cls.PROC_DIR
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
