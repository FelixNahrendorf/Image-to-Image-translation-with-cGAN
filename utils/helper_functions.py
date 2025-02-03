import os
import yaml

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return _set_dataset(config)

def _set_dataset(config):
    dataset_nr = config['dataset']
    datasets_dir = config['data']['datasets_dir']
    config['data']['train_images_dir'] = os.path.join(f"{datasets_dir}{dataset_nr}", config['data']['train_images_dir'])
    config['data']['train_labels_dir'] = os.path.join(f"{datasets_dir}{dataset_nr}", config['data']['train_labels_dir'])
    config['data']['test_images_dir'] = os.path.join(f"{datasets_dir}{dataset_nr}", config['data']['test_images_dir'])
    config['data']['test_labels_dir'] = os.path.join(f"{datasets_dir}{dataset_nr}", config['data']['test_labels_dir'])
    config['data']['val_images_dir'] = os.path.join(f"{datasets_dir}{dataset_nr}", config['data']['val_images_dir'])
    config['data']['val_labels_dir'] = os.path.join(f"{datasets_dir}{dataset_nr}", config['data']['val_labels_dir'])
    return config
