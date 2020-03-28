import json

def get_param_dict(path):
    with open(path, 'r') as f:
        params_dict = json.load(f)

    return params_dict