import os
from box.exceptions import BoxValueError
from box import ConfigBox
from cnnClassifier import logger
import yaml
import json
import joblib
from pathlib import Path
from ensure import ensure_annotations
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    '''reads yaml file and return
    Args:
        path_to_yaml(str) : path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox : ConfigBox type       
    '''
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded succesfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml is empty")
    except Exception as e:
        raise e 

@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    '''create list of directories

    Args:
        path_to_directories : list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False   
      
    '''
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



            


