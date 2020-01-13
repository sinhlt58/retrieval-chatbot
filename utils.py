import json
import io
import os

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print (f"Created path {path}")

def write_json_data(path, data, sort_keys=False):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=sort_keys,
    )

def read_json_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)
