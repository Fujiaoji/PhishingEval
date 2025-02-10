import os
import time
import argparse
import csv
import pandas as pd
import yaml
import numpy as np
import cv2

from datetime import datetime
from train_ob.inference_ob import pred_rcnn, config_rcnn
from siamese import phishpedia_classifier_logo
from siamese import phishpedia_config
COLORS = {
    0: (255, 255, 0),  # logo
    1: (36, 255, 12),  # input
    2: (0, 255, 255),  # button
    3: (0, 0, 255),  # label
    4: (255, 0, 0)  # block
}

def vis(img_path, pred_boxes):
    '''
    Visualize rcnn predictions
    :param img_path: str
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return None
    '''

    check = cv2.imread(img_path)
    if pred_boxes is None or len(pred_boxes) == 0:
        print("Pred_boxes is None or the length of pred_boxes is 0")
        return check
    pred_boxes = pred_boxes.numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes

    # draw rectangle
    for j, box in enumerate(pred_boxes):
        if j == 0:
            cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLORS[0], 2)
        else:
            cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLORS[1], 2)

    return check


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

        # plotvis = vis(img_path, pred_boxes)
        
        if len(pred_boxes) == 0:
            phish_category = 0  # Report as benign

        # If at least one element is reported
        else:
            ######################## Step2: Siamese (logo matcher) ########################################
            tagBox = "tagBox1"
            pred_target, matched_coord, siamese_conf = phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                      domain_map_path=DOMAIN_MAP_PATH,
                                                                      model=SIAMESE_MODEL,
                                                                      logo_feat_list=LOGO_FEATS,
                                                                      file_name_list=LOGO_FILES,
                                                                      url=url,
                                                                      shot_path=img_path,
                                                                      ts=SIAMESE_THRE)
            # cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
            #         (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # cv2.imwrite(("predict.png"), plotvis)
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
                        default="result_{}".format(date),
                        help='Output results file name')
    # weights parameter
    parser.add_argument('-siamese_weights',
                        type=str, 
                        required=True,
                        help='Siamese model weights path')
    parser.add_argument('-targetlist',
                        type=str, 
                        required=True,
                        help='Targetlist folder path')
    
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--no_repeat', action='store_true')

    args = parser.parse_args()

    
    # parameters
    ELE_CFG_PATH = 'configs/faster_rcnn.yaml'
    ELE_WEIGHTS_PATH = 'models/model_final.pth'
    ELE_CONFIG_THRE = 0.05
    ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)
    # siamese model
    NUM_CLASSES = 277
    DOMAIN_MAP_PATH = 'models/domain_map.pkl'
    SIAMESE_THRE = 0.83
    
    SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = None, None, None
    SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config(
        num_classes=NUM_CLASSES,
        weights_path=args.siamese_weights,
        targetlist_path=args.targetlist)
    print('Finish loading protected logo list')
    
    phishpedia_eval(args, ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
    print(f"Finish Eval Time: {time.time()}, Duration: {time.time()-startTime}")
