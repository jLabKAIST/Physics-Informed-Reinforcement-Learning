import json
import pickle
from pathlib import Path


def load_eff_table(filename):
    if Path(filename).exists():
        ext = filename.split('.')[-1]
        with open(filename, 'rb') as f:
            if ext == 'pkl':
                eff_table = pickle.load(f)
            elif ext == 'json':
                eff_table = json.load(f)
            else:
                raise RuntimeError('Unknown extension')

    return eff_table

