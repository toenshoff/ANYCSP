import os
import json
from src.data.dataset import File_Dataset, Generator_Dataset
from src.data.generators import generator_dict


def exists(path):
    path = os.path.join(path, 'config.json')
    return os.path.exists(path)


def write_config(config, path):
    path = os.path.join(path, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def read_config(path):
    with open(path, 'r') as f:
        conf_dict = json.load(f)
    return conf_dict


def dataset_from_config(data_config, num_samples=1000):
    if 'FILES' in data_config:
        dataset = File_Dataset(**data_config['FILES'])
    else:
        generators = [generator_dict[name](**kwargs) for name, kwargs in data_config.items()]
        dataset = Generator_Dataset(generators, num_samples)
    return dataset
