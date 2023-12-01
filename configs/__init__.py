import os
import yaml


def load_yaml(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError("cannot find file \"%s\", aborting.. " % file)
    
    elif not file.endswith(".yaml"):
        raise FileNotFoundError("invalid extension in \"%s\", aborting.. " % file)
    
    try:
        data = yaml.safe_load(open(file, 'r'))
    except Exception as e:
        print("yaml exception occurred \"%s\"" % e) 
    
    return data