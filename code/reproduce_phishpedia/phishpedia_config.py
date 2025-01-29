# Global configuration
import os
import subprocess
from typing import Union
import yaml
import numpy as np
from train_ob.inference_ob import config_rcnn
from siamese import phishpedia_config, phishpedia_config_easy

def load_config(cfg_path: Union[str, None], reload_targetlist=False):
    with open(os.path.join(os.path.dirname(__file__), cfg_path)) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH']
    ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
    ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
    ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)
    SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = None, None, None, None
    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']
    
    SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config(
        num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
        weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
        targetlist_path=configs['SIAMESE_MODEL']['TARGETLIST_PATH'])
    print('Finish loading protected logo list')
    print("feature save to ", os.path.join(os.path.dirname(__file__), 'LOGO_FEATS'))
    np.save(os.path.join(os.path.dirname(__file__), 'LOGO_FEATS'), LOGO_FEATS)
    np.save(os.path.join(os.path.dirname(__file__), 'LOGO_FILES'), LOGO_FILES)
    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']
    
    return ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH



def load_config_crop_logo(cfg_path: Union[str, None], reload_targetlist=False):
    with open(os.path.join(os.path.dirname(__file__), cfg_path)) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH']
    ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
    ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
    ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)

    return ELE_MODEL
