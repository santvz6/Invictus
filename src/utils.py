import json
from pathlib import Path


def load_json(json_path: Path) -> dict:
    """
    Carga un archivo JSON desde la ruta especificada.
    
    Args:
        json_path: Ruta al archivo JSON.
        
    Returns:
        Diccionario con los datos cargados.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        dataset_dict = json.load(f)
    return dataset_dict
    

def save_json(json_dict: dict, save_path: Path) -> None:
    """
    Guarda un diccionario en formato JSON en la ruta especificada.
    
    Args:
        json_dict: Diccionario a guardar.
        save_path: Ruta donde se guardará el archivo JSON.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)
    