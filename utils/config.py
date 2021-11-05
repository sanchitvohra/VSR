import yaml

def load_config(config_path):
    f = open(config_path, 'r')
    data = f.read()
    config = yaml.load(data)
    return config