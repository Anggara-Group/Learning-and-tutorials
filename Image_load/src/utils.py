import json 
# Load the JSON configuration file
def load_paths(config_file='paths.json'):
    """Load path configuration from JSON file"""
    with open(config_file, 'r') as f:
        paths = json.load(f)
    return paths

def deep_update(dict1, dict2):
    """Update dictionaries"""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_update(dict1[key], value)
        else:
            dict1[key] = value