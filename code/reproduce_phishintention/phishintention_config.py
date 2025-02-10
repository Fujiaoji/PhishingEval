import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from CRP_Classifier.models import KNOWN_MODELS
from torch.backends import cudnn
from typing import Union
from collections import OrderedDict
from tqdm import tqdm
from OCR_Siamese.model_builder import ModelBuilder
from OCR_Siamese.labelmaps import get_vocabulary
from PIL import Image, ImageOps

from torchvision import transforms

def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    
    img = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')

    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

def ocr_main(image_path, model, height=None, width=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Evaluation
    model.eval()
    
    img = image_process(image_path)
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)
    
    # TODO: testing should be more clean. to be compatible with the lmdb-based testing, need to construct some meaningless variables.
    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    rec_targets = torch.IntTensor(1, 100).fill_(1)
    rec_targets[:,100-1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets.to(device)
    input_dict['rec_lengths'] = [100]
    
    with torch.no_grad():
        features, decoder_feat = model.features(input_dict)
    features = features.detach().cpu()
    decoder_feat = decoder_feat.detach().cpu()
    features = torch.mean(features, dim=1)
    
    return features

class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """
    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)

def l2_norm(x):
    '''L2 Normalization'''
    if len(x.shape):
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)

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

def credential_config(checkpoint, model_type='mixed'):
    '''
    Load credential classifier configurations
    :param checkpoint: classifier weights
    :param model_type: layout|screenshot|mixed|topo
    :return model: classifier
    '''
    # load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'screenshot':
        model = KNOWN_MODELS['BiT-M-R50x1'](head_size=2)
    elif model_type == 'layout':
        model = KNOWN_MODELS['FCMaxV2'](head_size=2)
    elif model_type == 'mixed':
        model = KNOWN_MODELS['BiT-M-R50x1V2'](head_size=2)
    elif model_type == 'topo':
        model = KNOWN_MODELS['BiT-M-R50x1V3'](head_size=2)
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

def ocr_model_config(checkpoint, height=None, width=None):
    
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
  
    # Create data loaders
    if height is None or width is None:
        height, width = (32, 100)

    dataset_info = DataInfo('ALLCASES_SYMBOLS')

    # Create model
    model = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=dataset_info.rec_num_classes,
                         sDim=512, attDim=512, max_len_labels=100,
                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)

    # Load from checkpoint
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if device == 'cuda':
        model = model.to(device)
        
    return model

def pred_siamese_OCR(img, model, ocr_model, imshow=False, title=None, grayscale=False):
    '''
    Inference for a single image with OCR enhanced model
    :param img_path: image path in str or image in PIL.Image
    :param model: Siamese model to make inference
    :param ocr_model: pretrained OCR model
    :param imshow: enable display of image or not
    :param title: title of displayed image
    :param grayscale: convert image to grayscale or not
    :return feature embedding of shape (2048,)
    '''
    img_size = 224
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
        ])
    
    img = Image.open(img) if isinstance(img, str) else img
    img = img.convert("RGBA").convert("L").convert("RGB") if grayscale else img.convert("RGBA").convert("RGB")

    ## Resize the image while keeping the original aspect ratio
    pad_color = 255 if grayscale else (255, 255, 255)
    img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2, 
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=pad_color)     
 
    img = img.resize((img_size, img_size))
    
    ## Plot the image    
    if imshow: 
        if grayscale:
            plt.imshow(np.asarray(img), cmap='gray')
        else:
            plt.imshow(np.asarray(img))
        plt.title(title)
        plt.show()   
        
            
    with torch.no_grad():
        # get ocr embedding from pretrained paddleOCR
        ocr_emb = ocr_main(image_path=img, model=ocr_model, height=None, width=None)
        ocr_emb = ocr_emb[0]
        ocr_emb = ocr_emb[None, ...].to(device) # remove batch dimension
        
    # Predict the embedding
    with torch.no_grad():
        img = img_transforms(img)
        img = img[None, ...].to(device)
        logo_feat = model.features(img, ocr_emb)
        logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy() # L2-normalization final shape is (2560,)
        
    return logo_feat

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
    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
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
    
    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'): # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith('.jpg') or logo_path.endswith('.PNG') or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'): # skip homepage/loginpage
                    continue
                logo_feat_list.append(pred_siamese_OCR(img=os.path.join(targetlist_path, target, logo_path), 
                                                       model=model, ocr_model=ocr_model,
                                                       grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))
        
    return model, ocr_model, np.asarray(logo_feat_list), np.asarray(file_name_list)


def driver_loader():
    '''
    load chrome driver
    :return:
    '''

    options = initialize_chrome_settings(lang_txt=os.path.join(os.path.dirname(__file__), 'src/util/lang.txt'))
    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert
    capabilities["pageLoadStrategy"] = "eager"  # eager mode #FIXME: set eager mode, may load partial webpage

    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(),
                              desired_capabilities=capabilities,
                              chrome_options=options)
    driver.set_page_load_timeout(60)  # set timeout to avoid wasting time
    driver.set_script_timeout(60)  # set timeout to avoid wasting time
    helium.set_driver(driver)
    return driver

def login_config(rcnn_weights_path: str, rcnn_cfg_path: str, threshold=0.05, device='cuda'):
    '''
    Load login button detector configurations
    :param rcnn_weights_path: path to rcnn weights
    :param rcnn_cfg_path: path to configuration file
    :return cfg: rcnn cfg
    :return model: rcnn model
    '''
    # merge configuration
    cfg = get_cfg()
    cfg.merge_from_file(rcnn_cfg_path)
    cfg.MODEL.WEIGHTS = rcnn_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # lower this threshold to report more boxes
    if device == 'cpu':
        cfg.MODEL.DEVICE = 'cpu'
    
    # initialize model
    model = DefaultPredictor(cfg)
    return cfg, model

def load_config(cfg_path: Union[str, None] = None, reload_targetlist=True, device='cuda'):

    #################### '''Default''' ####################
    if cfg_path is None:
        print("XXXX", os.path.join(os.path.dirname(__file__), 'configs.yaml'))
        with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(cfg_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

    # element recognition model
    AWL_CFG_PATH = configs['AWL_MODEL']['CFG_PATH']
    AWL_WEIGHTS_PATH = configs['AWL_MODEL']['WEIGHTS_PATH']
    AWL_CONFIG, AWL_MODEL = element_config(rcnn_weights_path=AWL_WEIGHTS_PATH,
                                           rcnn_cfg_path=AWL_CFG_PATH, device=device)

    CRP_CLASSIFIER = credential_config(
        checkpoint=configs['CRP_CLASSIFIER']['WEIGHTS_PATH'],
        model_type=configs['CRP_CLASSIFIER']['MODEL_TYPE'])

    CRP_LOCATOR_CONFIG, CRP_LOCATOR_MODEL = login_config(
        rcnn_weights_path=configs['CRP_LOCATOR']['WEIGHTS_PATH'],
        rcnn_cfg_path=configs['CRP_LOCATOR']['CFG_PATH'],
        device=device)

    # siamese model
    print('Load protected logo list')
    
    print(configs['SIAMESE_MODEL']['NUM_CLASSES'], configs['SIAMESE_MODEL']['WEIGHTS_PATH'], configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'], configs['SIAMESE_MODEL']['TARGETLIST_PATH'])
    
    SIAMESE_MODEL, OCR_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config_OCR(
        num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
        weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
        ocr_weights_path=configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'],
        targetlist_path=configs['SIAMESE_MODEL']['TARGETLIST_PATH'])
    np.save(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FEATS'), LOGO_FEATS)
    np.save(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FILES'), LOGO_FILES)

    print('Finish loading protected logo list')

    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']  # FIXME: threshold is 0.87 in phish-discovery?

    # brand-domain dictionary
    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH #CRP_LOCATOR_MODEL

