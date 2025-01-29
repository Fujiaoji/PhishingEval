import os
import time
import argparse
import csv
import pandas as pd

from datetime import datetime
from phishpedia_config import load_config
from train_ob.inference_ob import pred_rcnn
from siamese import phishpedia_classifier_logo

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def phishpedia_eval(args, ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):
    # csv contains the results
    normal_csv = open(args.output_csv + "_" + str(SIAMESE_THRE) + ".csv", "w")
    normal_csvwriter = csv.writer(normal_csv)
    normal_csvwriter.writerow(["scr_path", "true_brand", "pred_brand", "phish", "tagBox", "siamese_conf", "url"])
  
    csv_data = pd.read_csv(args.input_csv)
    
    for idx, row in csv_data.iterrows():
        tagBox = "tagBox0"
    
        phish_category = 0  # 0 for benign, 1 for phish
        pred_target = None  # predicted target, default is None
        url = row["domain"]
        siamese_conf = 0
        img_path = row["scr_path"]

        if not os.path.exists(img_path):  # screenshot not exist
            print("{}: {} screenshot not exist".format(idx, img_path))
            continue
        
        
        ####################### Step1: layout detector ##############################################
        # detectron2_pedia.inference
        pred_boxes, _, _, _ = pred_rcnn(im=img_path, predictor=ELE_MODEL)
        pred_boxes = pred_boxes.detach().cpu().numpy()

        if len(pred_boxes) == 0:
            phish_category = 0  # Report as benign

        # If at least one element is reported
        else:
            ######################## Step2: Siamese (logo matcher) ########################################
            tagBox = "tagBox1"
            pred_target, _, siamese_conf = phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                      domain_map_path=DOMAIN_MAP_PATH,
                                                                      model=SIAMESE_MODEL,
                                                                      logo_feat_list=LOGO_FEATS,
                                                                      file_name_list=LOGO_FILES,
                                                                      url=url,
                                                                      shot_path=img_path,
                                                                      ts=SIAMESE_THRE)
            # Phishpedia reports target
            if pred_target is not None:
                phish_category = 1  # Report as suspicious

            # Phishpedia does not report target
            else:  # Report as benign
                phish_category = 0

        try:
            normal_csvwriter.writerow([row["scr_path"], row["brand"], pred_target, str(phish_category), tagBox, str(siamese_conf), url])
        except:
            normal_csvwriter.writerow([row["scr_path"], row["brand"], pred_target, str(phish_category), tagBox, str(siamese_conf), None])



if __name__ == '__main__':
    date = datetime.today().strftime('%Y-%m-%d')
    print('Today is:', date)
    start_time = datetime.now()
    startTime = time.time()
    print(f"Start Eval Time: {startTime}")
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--input_csv",
                        default="data_test/data_test.csv",
                        help='Input csv path to test')
    
    parser.add_argument('-r', "--output_csv", 
                        default="results/result_{}".format(date),
                        help='Output results file name')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--no_repeat', action='store_true')

    args = parser.parse_args()

    
    # 22
    # ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH']
    # ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
    # ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
    # ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)
    # SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = None, None, None, None
    # # siamese model
    # SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']
    
    # SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config(
    #     num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
    #     weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
    #     targetlist_path=configs['SIAMESE_MODEL']['TARGETLIST_PATH'])
    # print('Finish loading protected logo list')
    # print("feature save to ", os.path.join(os.path.dirname(__file__), 'LOGO_FEATS'))
    # np.save(os.path.join(os.path.dirname(__file__), 'LOGO_FEATS'), LOGO_FEATS)
    # np.save(os.path.join(os.path.dirname(__file__), 'LOGO_FILES'), LOGO_FILES)
    # DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']






    ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(cfg_path="configs.yaml", reload_targetlist=True)
    # print(ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
    phishpedia_eval(args, ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
    print(f"Finish Eval Time: {time.time()}, Duration: {time.time()-startTime}")
