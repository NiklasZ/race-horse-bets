import json


def get_training_config(file_path: str) -> dict:
    with open(file_path) as f:
        return json.load(f)
