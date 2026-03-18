"""
Mapeo ponderado de Barrios AMAEM → Municipios INE.

Estructura: {barrio: [(codigo_ine, peso), ...]}
  - codigo_ine: string con el código del municipio (sin nombre), ej. '03014'
  - peso: float en [0.0, 1.0] que indica qué fracción del barrio pertenece a ese municipio.
  - Los pesos de un barrio suman <= 1.0 (la fracción restante se considera sin cobertura/NaN).

"""

import os
import sys
import yaml
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import Paths


BARRIO_MUNICIPIO_WEIGHTS: dict[str, list[tuple[str, float]]] = { ... }

def export_yaml_to_csv():
    yaml_path = Paths.MAPPING_BARRIOS_YAML
    csv_path = Paths.MAPPING_BARRIOS

    if not os.path.exists(yaml_path):
        print(f"No existe el archivo YAML en {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []
    for barrio, asignaciones in data.items():
        for item in asignaciones:
            muni, peso = item
            rows.append({
                "barrio": barrio,
                "municipio": muni,
                "peso": peso
            })
           
    df_map = pd.DataFrame(rows)
    df_map.to_csv(csv_path, index=False, sep=";")

if __name__ == "__main__":
    export_yaml_to_csv()