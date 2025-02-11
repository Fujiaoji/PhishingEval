import os
import csv
import time
import argparse
import pandas as pd

from datetime import datetime
from PIL import Image

from config import load_config
from inference_ob import pred_rcnn
from inference_ob import config_rcnn



os.environ['KMP_DUPLICATE_LIB_OK']='True'


def load_config():
    ELE_CFG_PATH = "faster_rcnn.yaml"
    ELE_WEIGHTS_PATH = "output/model_final.pth"
    ELE_CONFIG_THRE = 0.05

    ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)

    return ELE_MODEL


def run_crop(args, ELE_MODEL):
    
    start_time = time.time()
    df = pd.read_csv(args.input_folder)
    crop_image_path = []
    iii = 0
    
    for index, row in df.iterrows():
        img_path = os.path.join("/dataset/fujiao/baseline_dataset/class540/crawl_dataset/phishing_dataset", row["screenshot_path"])
        des_path = img_path.replace("shot.png", "crop_logo.png")
        ####################### Step1: layout detector ##############################################
        # detectron2_pedia.inference
        pred_boxes, pred_box_scores, _, _ = pred_rcnn(im=img_path, predictor=ELE_MODEL)
        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_box_scores = pred_box_scores.detach().cpu().numpy()
        if len(pred_boxes) == 0:
            print("no pred box", img_path)
        else:
            min_x, min_y, max_x, max_y = pred_boxes[0]
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            try:
                img = Image.open(img_path)
            except OSError:  # if the image cannot be identified, return nothing
                print('Screenshot cannot be open')
                return None, None, None
            cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            cropped.save(des_path)
            crop_image_path.append(des_path)
            iii += 1
                
        with open("crop_"+ args.brand.lower() + "_phishing.txt", "w") as f:
            for ctem in crop_image_path:
                # 将每个元素写入文件，每个元素后加上换行符
                f.write(ctem + '\n')
    end_time = time.time() - start_time
    print("use {} time".format(end_time))



if __name__ == '__main__':
    date = datetime.today().strftime('%Y-%m-%d')
    print('Today is:', date)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', "--input_csv",
                        default="/dataset/fujiao/baseline_dataset/class540/crawl_dataset/phishing_dataset/google_phishing.csv",
                        help='Input folder path to parse')
    parser.add_argument('-b', "--brand",
                        default="google",
                        help='Input folder path to parse')
    parser.add_argument('-pb', "--pb",
                        default="phishing",
                        help='Input folder path to parse')
    
    args = parser.parse_args()
    
    ELE_MODEL = load_config()
    
    run_crop(args, ELE_MODEL)
    print('Process finish')
    

