import os
import csv
import time
import argparse
import pandas as pd
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transform
import pickle
import tldextract

from datetime import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from CRP_Classifier.models import KNOWN_MODELS
from collections import OrderedDict
from OCR_Siamese.demo import ocr_model_config
from tqdm import tqdm
from OCR_Siamese.inference import pred_siamese_OCR, siamese_inference_OCR
from CRP_Classifier.HTML_heuristic.post_form import *
from PIL import Image
from CRP_Classifier.grid_divider import coord2pixel_reverse
from os.path import join as pjoin


def element_config(rcnn_weights_path: str, rcnn_cfg_path: str, device='cuda'):
    '''
    Load element detector configurations
    :param rcnn_weights_path: path to rcnn weights
    :param rcnn_cfg_path: path to configuration file
    :return cfg: rcnn cfg
    :return model: rcnn model
    '''
    # merge configuration
    cfg = get_cfg()
    cfg.merge_from_file(rcnn_cfg_path)
    cfg.MODEL.WEIGHTS = rcnn_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # lower this threshold to report more boxes
    if device == 'cpu':
        cfg.MODEL.DEVICE = 'cpu' # if you installed detectron2 for cpu version
    
    # initialize model
    model = DefaultPredictor(cfg)
    return cfg, model

def element_recognition(img, model):
    '''
    Recognize elements from a screenshot
    :param img: [str|np.ndarray]
    :param model: rcnn model
    :return pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :return pred_scores: torch.Tensor of shape Nx1, prediction confidence of bounding boxes
    '''
    if isinstance(img, str):
        img_processed = cv2.imread(img)
        if img_processed is not None:
            if img_processed.shape[-1] == 4:
                img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGRA2BGR)
        else:
            return None, None, None
    elif isinstance(img, np.ndarray):
        img_processed = img
    else:
        raise NotImplementedError

    pred = model(img_processed)
    pred_i = pred["instances"].to("cpu")
    pred_classes = pred_i.pred_classes # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor # Boxes coords
    pred_scores = pred_i.scores # Boxes prediction scores

    return pred_classes, pred_boxes, pred_scores


def html_heuristic(html_path):
    '''
    Call HTML heuristic
    :param html_path: path to html file
    :return: CRP = 0 or nonCRP = 1
    '''
    tree = read_html(html_path)
    proc_data = proc_tree(tree)
    return check_post(proc_data, version=2)

def credential_classifier_mixed_al(img:str, coords, types, model):
    '''
    Run credential classifier for AL dataset
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: CRP = 0 or nonCRP = 1
    :return conf: torch.Tensor NxC prediction confidence
    '''
    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(img).convert('RGB')
    
    # transform to tensor
    transformation = transform.Compose([transform.Resize((256, 512)), 
                                        transform.ToTensor()])
    image = transformation(image)
    
    # append class channels
    # class grid tensor is of shape 8xHxW
    grid_tensor = coord2pixel_reverse(img_path=img, coords=coords, types=types, reshaped_size=(256, 512))

    image = torch.cat((image.double(), grid_tensor), dim=0)
    assert image.shape == (8, 256, 512) # ensure correct shape

    # inference
    with torch.no_grad():
       
        pred_features = model.features(image[None,...].to(device, dtype=torch.float))
        pred_orig = model(image[None,...].to(device, dtype=torch.float))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
    return pred, conf, pred_features

def credential_config(checkpoint, model_type='mixed'):
    '''
    Load credential classifier configurations
    :param checkpoint: classifier weights
    :param model_type: layout|screenshot|mixed|topo
    :return model: classifier
    '''
    # load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'mixed':
        model = KNOWN_MODELS['BiT-M-R50x1V2'](head_size=2)
    else:
        raise ValueError('CRP Model type not supported, please use one of the following [screenshot|layout|mixed|topo]')
        
    checkpoint = torch.load(checkpoint, map_location="cpu")
    checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if not k.startswith('module'):
            new_state_dict[k]=v
            continue
        name = k.split('module.')[1]
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def phishpedia_classifier_OCR(pred_classes, pred_boxes, 
                          domain_map_path:str,
                          model, ocr_model, logo_feat_list, file_name_list, shot_path:str, 
                          url:str, 
                          ts:float):
    '''
    Run siamese
    :param pred_classes: torch.Tensor/np.ndarray Nx1 predicted box types
    :param pred_boxes: torch.Tensor/np.ndarray Nx4 predicted box coords
    :param domain_map_path: path to domain map dict
    :param model: siamese model
    :param ocr_model: ocr model
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
        
    # look at boxes for logo class only
    logo_boxes = pred_boxes[pred_classes==0] 
    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None
   
    print(f"---len box {len(logo_boxes)}")

    # run logo matcher
    # pred_target = None
    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = siamese_inference_OCR(model, ocr_model, domain_map, 
                                                         logo_feat_list, file_name_list,
                                                         shot_path, bbox, t_s=ts, grayscale=False)
            
            # domain matcher to avoid FP
            if matched_target is not None:
                matched_coord = coord
                # if tldextract.extract(url).domain+ '.'+tldextract.extract(url).suffix not in matched_domain:
                if tldextract.extract(url).domain not in matched_domain:
                    # avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                    if matched_target == 'GoDaddy' or matched_target == "Webmail Provider" or matched_target == "Government of the United Kingdom":
                        matched_target = None # ignore the prediction
                        matched_domain = None # ignore the prediction
                else: # benign, real target
                    matched_target = None  # ignore the prediction
                    matched_domain = None  # ignore the prediction
                break # break if target is matched
            break # only look at 1st logo

    return brand_converter(matched_target), matched_coord, this_conf


def phishpedia_config_OCR(num_classes:int, weights_path:str, 
                          ocr_weights_path:str,
                          targetlist_path:str, grayscale=False):
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
    
    # load OCR model
    ocr_model = ocr_model_config(checkpoint=ocr_weights_path)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from OCR_Siamese.models import KNOWN_MODELS
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)


    # # Load weights
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        # New task might have different classes; remove the pretrained classifier weights
        del checkpoint['model']['additionalfc.conv_add.weight']
        del checkpoint['model']['additionalfc.conv_add.bias']
        model.load_state_dict(checkpoint["model"], strict=False)
    
    # Load weights
    weights = torch.load(weights_path, map_location=device)
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if k.startswith('module'):
            name = k.split('module.')[1]
        else:
            name = k
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
#     Prediction for targetlists
    logo_feat_list = []
    file_name_list = []
    files_path = os.listdir(targetlist_path)
    files_path = [item for item in files_path if (not item.startswith(".")) and (not item.endswith(".npy"))]
    for target in tqdm(files_path):
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith(('.png','.jpeg', '.jpg', '.PNG','.JPG', '.JPEG')) and (not logo_path.startswith(('loginpage', 'homepage'))):
                logo_feat_list.append(pred_siamese_OCR(img=os.path.join(targetlist_path, target, logo_path), 
                                                       model=model, ocr_model=ocr_model,
                                                       grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))
        
    return model, ocr_model, np.asarray(logo_feat_list), np.asarray(file_name_list)


def brand_converter(brand_name):
    '''
    Helper function to deal with inconsistency in brand naming
    '''
    if brand_name == 'Adobe Inc.' or brand_name == 'Adobe Inc':
        return 'Adobe'
    elif brand_name == 'ADP, LLC' or brand_name == 'ADP, LLC.':
        return 'ADP'
    elif brand_name == 'Amazon.com Inc.' or brand_name == 'Amazon.com Inc':
        return 'Amazon'
    elif brand_name == 'Americanas.com S,A Comercio Electrnico':
        return 'Americanas.com S'
    elif brand_name == 'AOL Inc.' or brand_name == 'AOL Inc':
        return 'AOL'
    elif brand_name == 'Apple Inc.' or brand_name == 'Apple Inc':
        return 'Apple'
    elif brand_name == 'AT&T Inc.' or brand_name == 'AT&T Inc':
        return 'AT&T'
    elif brand_name == 'Banco do Brasil S.A.':
        return 'Banco do Brasil S.A'
    elif brand_name == 'Credit Agricole S.A.':
        return 'Credit Agricole S.A'
    elif brand_name == 'DGI (French Tax Authority)':
        return 'DGI French Tax Authority'
    elif brand_name == 'DHL Airways, Inc.' or brand_name == 'DHL Airways, Inc' or brand_name == 'DHL':
        return 'DHL Airways'
    elif brand_name == 'Dropbox, Inc.' or brand_name == 'Dropbox, Inc':
        return 'Dropbox'
    elif brand_name == 'eBay Inc.' or brand_name == 'eBay Inc':
        return 'eBay'
    elif brand_name == 'Facebook, Inc.' or brand_name == 'Facebook, Inc':
        return 'Facebook'
    elif brand_name == 'Free (ISP)':
        return 'Free ISP'
    elif brand_name == 'Google Inc.' or brand_name == 'Google Inc':
        return 'Google'
    elif brand_name == 'Mastercard International Incorporated':
        return 'Mastercard International'
    elif brand_name == 'Netflix Inc.' or brand_name == 'Netflix Inc':
        return 'Netflix'
    elif brand_name == 'PayPal Inc.' or brand_name == 'PayPal Inc':
        return 'PayPal'
    elif brand_name == 'Royal KPN N.V.':
        return 'Royal KPN N.V'
    elif brand_name == 'SF Express Co.':
        return 'SF Express Co'
    elif brand_name == 'SNS Bank N.V.':
        return 'SNS Bank N.V'
    elif brand_name == 'Square, Inc.' or brand_name == 'Square, Inc':
        return 'Square'
    elif brand_name == 'Webmail Providers':
        return 'Webmail Provider'
    elif brand_name == 'Yahoo! Inc' or brand_name == 'Yahoo! Inc.':
        return 'Yahoo!'
    elif brand_name == 'Microsoft OneDrive' or brand_name == 'Office365' or brand_name == 'Outlook':
        return 'Microsoft'
    elif brand_name == 'Global Sources (HK)':
        return 'Global Sources HK'
    elif brand_name == 'T-Online':
        return 'Deutsche Telekom'
    elif brand_name == 'Airbnb, Inc':
        return 'Airbnb, Inc.'
    elif brand_name == 'azul':
        return 'Azul'
    elif brand_name == 'Raiffeisen Bank S.A':
        return 'Raiffeisen Bank S.A.'
    elif brand_name == 'Twitter, Inc' or brand_name == 'Twitter':
        return 'Twitter, Inc.'
    elif brand_name == 'capital_one':
        return 'Capital One Financial Corporation'
    elif brand_name == 'la_banque_postale':
        return 'La Banque postale'
    elif brand_name == 'db':
        return 'Deutsche Bank AG'
    elif brand_name == 'Swiss Post' or brand_name == 'PostFinance':
        return 'PostFinance'
    elif brand_name == 'grupo_bancolombia':
        return 'Bancolombia'
    elif brand_name == 'barclays':
        return 'Barclays Bank Plc'
    elif brand_name == 'gov_uk':
        return 'Government of the United Kingdom'
    elif brand_name == 'Aruba S.p.A':
        return 'Aruba S.p.A.'
    elif brand_name == 'TSB Bank Plc':
        return 'TSB Bank Limited'
    elif brand_name == 'strato':
        return 'Strato AG'
    elif brand_name == 'cogeco':
        return 'Cogeco'
    elif brand_name == 'Canada Revenue Agency':
        return 'Government of Canada'
    elif brand_name == 'UniCredit Bulbank':
        return 'UniCredit Bank Aktiengesellschaft'
    elif brand_name == 'ameli_fr':
        return 'French Health Insurance'
    elif brand_name == 'Banco de Credito del Peru':
        return 'bcp'
    else:
        return brand_name

def phishintention_eval(args, siamese_ts):
    '''
    Run phishintention evaluation
    :param data_dir: data folder dir
    :param mode: phish|benign
    :param siamese_ts: siamese threshold
    :param write_txt: txt path to write results
    :return:
    '''
    normal_csv = open(args.output_csv, "w")
    normal_csvwriter = csv.writer(normal_csv)
    normal_csvwriter.writerow(["scr_path", "domain", "true_brand", "pred_brand", "phish", "siamese_conf", "url", "tagBox", "tagCRPHTML", "tagCRPscreenshot", "tagBrand"])# tag=tag0=None, tag=tag1=Have
    # testing url, need to change column name based on the csv file column name
    df = pd.read_csv(args.input_csv)
    
    start_time = time.time()
    for idx, row in df.iterrows():
        phish_category = 0  # 0 for benign, 1 for phish
        pred_target = None  # predicted target, default is None
        siamese_conf = 0
        
        tagBox = "tagBox0"
        tagCRPHTML = "tagCRPHTML0"
        tagCRPscreenshot = "tagCRPscreenshot0"
        tagBrand = "tagBrand0"
        
        url = row["domain"]
        img_path = row["scr_path"]
        html_path = img_path.replace(".png", ".html")
        
        # Element recognition module
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0  # Report as benign

        # If at least one element is reported
        else:
            tagBox = "tagBox1"
            # Credential heuristic module
            cred_conf = None
            # CRP HTML heuristic
            cre_pred = html_heuristic(html_path)
            # Credential classifier module
            if cre_pred == 1:  # if HTML heuristic report as nonCRP
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=img_path, 
                                                                        coords=pred_boxes,
                                                                        types=pred_classes, 
                                                                        model=cls_model)
            else: 
                tagCRPHTML = "tagCRPHTML1"
                

            # Non-credential page
            if cre_pred == 1:  # if non-CRP
                phish_category = 0  # Report as benign

            # Credential page
            else:
                # Phishpedia module
                tagCRPscreenshot = "tagCRPscreenshot1"
                pred_target, _, siamese_conf = phishpedia_classifier_OCR(pred_classes=pred_classes, 
                                                                         pred_boxes=pred_boxes,
                                                                         domain_map_path=domain_map_path,
                                                                         model=pedia_model,
                                                                         ocr_model=ocr_model,
                                                                         logo_feat_list=logo_feat_list, 
                                                                         file_name_list=file_name_list,
                                                                         url=url,
                                                                         shot_path=img_path,
                                                                         ts=siamese_ts)


                # Phishpedia reports target
                if pred_target is not None:
                    phish_category = 1  # Report as suspicious
                    tagBrand = "tagBrand1"
                # Phishpedia does not report target
                else:  # Report as benign
                    phish_category = 0
        pred_brand = brand_converter(pred_target) if pred_target is not None else None
        normal_csvwriter.writerow([row["scr_path"], row["domain"], row["brand"], pred_brand, str(phish_category), str(siamese_conf), url, tagBox, tagCRPHTML, tagCRPscreenshot, tagBrand])



if __name__ == '__main__':
    date = datetime.today().strftime('%Y-%m-%d')
    print('Today is:', date)
    start_time = datetime.now()
    startTime = time.time()
    print(f"Start Eval Time: {startTime}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--input_csv",
                        default="data_test/data_test.csv",
                        help='Input dataset csv file')
    parser.add_argument('-r', 
                        "--output_csv", default="eval_result_{}.csv".format(date),
                        help='Output results csv')
    
    parser.add_argument('-new', "--expand", required=True, type=str,
                        help='Y=expand277_new, N=expand277')
    
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--no_repeat', action='store_true')

    args = parser.parse_args()
    
    rcnn_weights_path = 'models/model_final.pth'                   
    rcnn_cfg_path = "AWL/configs/faster_rcnn_web.yaml"
    checkpoint = "models/BiT-M-R50x1V2_0.005.pth.tar"
    ocr_weights_path = "models/demo.pth.tar"
    
    if args.expand=="Y": # expand277_new
        weights_path = "models/bit_new.pth.tar"
        targetlist_path = '../../data/targetlist/expand277_new'
    
    else: # expand277
        weights_path = "models/bit.pth.tar"
        targetlist_path = '../../data/targetlist/expand277'
    domain_map_path = 'models/domain_map.pkl'
    
    ele_cfg, ele_model = element_config(rcnn_weights_path=rcnn_weights_path, rcnn_cfg_path=rcnn_cfg_path)

    args.ele_model = ele_model
    args.ele_cfg = ele_cfg

    cls_model = credential_config(checkpoint=checkpoint, model_type='mixed')
    
    pedia_model, ocr_model, logo_feat_list, file_name_list = phishpedia_config_OCR(num_classes=277,
                                                                                   weights_path=weights_path, 
                                                                                   ocr_weights_path=ocr_weights_path,
                                                                                   targetlist_path=targetlist_path)
    print('Number of protected logos = {}'.format(str(len(logo_feat_list))))

    phishintention_eval(args, siamese_ts=0.83)
    print(f"Finish Eval Time: {time.time()}, Duration: {time.time() - startTime}")


            
 



