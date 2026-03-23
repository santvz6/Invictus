"""
Mapeo ponderado de Municipios INE a Barrios AMAEM (Alicante).

Este archivo gestiona la distribución de métricas estadísticas (viviendas turísticas,
población, etc.) desde los municipios del INE hacia los barrios de suministro.
Estructura: {nombre_barrio: [(codigo_ine, peso), ...]}
  - codigo_ine: Identificador del municipio del que se extraen los datos (ej. '03014').
  - peso: Proporción de las métricas del municipio que se asignan al barrio (0.0 a 1.0).
  - Nota: Permite calcular estimaciones de viviendas turísticas o features del INE por barrio.
"""


import os
import sys
import yaml
import pandas as pd

# Ajuste de ruta para permitir importaciones desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import Paths


def export_yaml_to_csv():
    """
    Convierte la configuración de mapeo de formato YAML a CSV.
    Útil para integraciones con herramientas externas o inspección rápida de datos.
    """
    yaml_path = Paths.MAPPING_BARRIOS_YAML
    csv_path = Paths.MAPPING_BARRIOS

    if not os.path.exists(yaml_path):
        print(f"Error: El archivo fuente YAML no se encuentra en {yaml_path}")
        return

    # Carga de datos estructurados desde YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Aplanamiento de la estructura para formato tabular
    rows = []
    for barrio, asignaciones in data.items():
        for item in asignaciones:
            muni, peso = item
            rows.append({
                "barrio": barrio,
                "municipio": muni,
                "peso": peso
            })
           
    # Generación y guardado del DataFrame resultante
    df_map = pd.DataFrame(rows)
    df_map.to_csv(csv_path, index=False, sep=";")
    print(f"Mapeo exportado correctamente a: {csv_path}")

if __name__ == "__main__":
    export_yaml_to_csv()