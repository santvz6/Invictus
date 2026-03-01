import os
import shutil

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent # ../src/config/paths.py (3 levels)


class Paths:
    ROOT = BASE_DIR
    SRC = ROOT / "src"
    INTERNAL = ROOT / "internal"
   
    DATA_DIR = INTERNAL / "data"   
    EXPERIMENTS_DIR = INTERNAL / "experiments"
    LOGS_DIR = INTERNAL / "logs"
    TEMP_DIR = INTERNAL / "temp"
    PROCESSED_DIR = INTERNAL / "processed" 



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


