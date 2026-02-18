import json
from pathlib import Path

def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        dataset_dict = json.load(f)
    return dataset_dict
    

def save_json(json_dict: dict, save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)
    