import json
import ruamel_yaml
import numpy as np 

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def parse_yaml(file='config/default.yaml'):
    with open(file) as f:
        return ruamel_yaml.load(f, Loader=ruamel_yaml.Loader)

def format_config(config, indent=2):
    return json.dumps(config, indent=indent, cls=NpEncoder)

def write_json(ctx, path, verbose=False):
    with open(path, "w") as f:
        json_string = json.dumps(ctx, indent=4)
        f.write(json_string)
