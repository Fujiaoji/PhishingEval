import os
import torch
import pickle
import tldextract
import numpy as np

from tqdm import tqdm
from collections import OrderedDict

from models import KNOWN_MODELS
from train_siamese.utils import brand_converter
from train_siamese.inference_siamese import siamese_inference, pred_siamese

def phishpedia_config(num_classes:int, weights_path:str, targetlist_path:str, grayscale=False):
    '''
    Load phishpedia configurations
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :param targetlist_path: targetlist folder
    :param grayscale: convert logo to grayscale or not, default is RGB
    :return model: siamese model
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"use {device} to run")
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location=device)
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    
    for k, v in weights.items():
        name = k.split('module.')[1]
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Prediction for targetlists
    logo_feat_list = []
    file_name_list = []
    files_list = os.listdir(targetlist_path)
    files_list = [item for item in files_list if (not item.startswith(".")) and (not item.endswith((".npy", ".csv", ".txt")))]
    for target in tqdm(files_list):
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith('.jpg') or logo_path.endswith('.PNG') \
                                          or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'): # skip homepage/loginpage
                    continue
                logo_feat_list.append(pred_siamese(img=os.path.join(targetlist_path, target, logo_path), 
                                                   model=model, grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))
    print(len(logo_feat_list), len(file_name_list))
    return model, np.asarray(logo_feat_list), np.asarray(file_name_list)


def phishpedia_config_easy(num_classes: int, weights_path: str):
    '''
    Load phishpedia model only
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :param targetlist_path: targetlist folder
    :param grayscale: convert logo to grayscale or not, default is RGB
    :return model: siamese model
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location=device)
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.split('module.')[1]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


def phishpedia_classifier_logo(logo_boxes,
                          domain_map_path: str,
                          model, logo_feat_list, file_name_list, shot_path: str,
                          url: str,
                          ts: float):
    '''
    Run siamese
    :param logo_boxes: torch.Tensor/np.ndarray Nx4 logo box coords
    :param domain_map_path: path to domain map dict
    :param model: siamese model
    :param logo_feat_list: targetlist embeddings
    :param file_name_list: targetlist paths
    :param shot_path: path to image
    :param url: url
    :param ts: siamese threshold
    :return pred_target
    :return coord: coordinate for matched logo
    '''
    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)
        
    
    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None
    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = siamese_inference(model, 
                                                                          domain_map,
                                                                          logo_feat_list,
                                                                          file_name_list,
                                                                          shot_path,
                                                                          bbox,
                                                                          t_s=ts, 
                                                                          grayscale=False)
            # domain matcher to avoid FP
            if matched_target is not None:
                matched_coord = coord
                if tldextract.extract(url).domain not in matched_domain:
                    # FIXME: avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                    if matched_target == 'GoDaddy' or matched_target == "Webmail Provider" or matched_target == "Government of the United Kingdom":
                        matched_target = None  # ignore the prediction
                        matched_domain = None
                else:# benign
                    matched_target = None
                    matched_domain = None
                
                break  # break if target is matched
            if i >= 2:  # only look at top-2 logo
                break
    return brand_converter(matched_target), matched_coord, this_conf

