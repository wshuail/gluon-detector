import os
import sys
import yaml


def get_config(name):
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)
    config = config[name.upper()]

    return config




