import os
import operator
import argparse
import csv
import time
import numpy as np
from math import sqrt
from PIL import Image
from datetime import datetime
from collections import Counter
from utils import brand_converter
import pandas as pd
import multiprocessing
from multiprocessing import Pool

# Define parameters
w = h = 100
s = 20
CDF = 32
p = q = 0.5


class Emd:#自定义的元素
    def __init__(self,emd,targetlist_name):
        self.emd = emd
        self.targetlist_name = targetlist_name

def get_signature(path):
    img = Image.open(path)
    img = img.resize((w, h), Image.LANCZOS)
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        img = img.convert("RGBA")
        r, g, b, a = img.split()
    # RGBA
    RGBA = []
    pixel = []
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel.append((i, j))
            RGBA.append((r.getpixel((i, j)) // CDF,
                         g.getpixel((i, j)) // CDF,
                         b.getpixel((i, j)) // CDF,
                         a.getpixel((i, j)) // CDF))
    # Centroid
    Ss = Counter(RGBA).most_common(s)
    signature = []
    # max_rgba
    r = []
    g = []
    b = []
    a = []
    for item in Ss:
        Cdcx = 0
        Cdcy = 0
        dc = item[0]
        r.append(dc[0])
        g.append(dc[1])
        b.append(dc[2])
        a.append(dc[3])
        for i, rgba in enumerate(RGBA):
            if rgba == dc:
                Cdcx += pixel[i][0]
                Cdcy += pixel[i][1]
        Cdc = (Cdcx/item[1], Cdcy/item[1])
        Fdc = (dc, Cdc)
        signature.append((Fdc, item[1]))
    md_color = sqrt(pow(max(r), 2)+pow(max(g), 2)+pow(max(b), 2)+pow(max(a), 2))
    return signature, md_color

def get_feature(signatureA, signatureB, md_colorA, md_colorB):
    md_color = max(md_colorA, md_colorB)
    md_centroid = sqrt(w*h)
    dis_color = np.zeros((s, s), dtype=float)
    dis_centroid = np.zeros((s, s), dtype=float)
    emd = 0
    for i, pixA in enumerate(signatureA):
        colorA = pixA[0][0]
        centroidA = pixA[0][1]
        for j, pixB in enumerate(signatureB):
            colorB = pixB[0][0]
            centroidB = pixB[0][1]
            color = (colorA[0]-colorB[0], colorA[1]-colorB[1], colorA[2]-colorB[2], colorA[3]-colorB[3])
            centroid = (centroidA[0]-centroidB[0], centroidA[1]-centroidB[1])
            dis_color[i][j] = sqrt(np.dot(color, color))
            dis_centroid[i][j] = sqrt(np.dot(centroid, centroid))
    dis_color /= md_color
    dis_centroid /= md_centroid
    dis = p*dis_color + q*dis_centroid
    for i in range(s):
        mind = np.min(dis[i], axis=0)
        ind = np.where(dis[i] == mind)
        dis = np.delete(dis, ind[0], axis=1)
        emd += mind
    emd /= s
    if emd > 0.3:
        emd *= 2
    elif emd < 0.3:
        emd /= 2
    if 1-emd < 0:
        return 0
    return 1 - emd


def get_targetlist_feature(target_path):
    sB_list = []
    mB_list = []
    tB_list = []
    targetlist_list = os.listdir(target_path)
    brand_targetlist = [item for item in targetlist_list if (not item.startswith(".")) and (not item.endswith((".txt", ".pkl", ".npy")))]
    print(f"---there are {len(brand_targetlist)} folders---")
    for brand in brand_targetlist:
        brand_pic_lists = os.listdir(os.path.join(target_path, brand))
        brand_pic_list = [os.path.join(target_path, brand, item) for item in brand_pic_lists if item.startswith("T")]
        if len(brand_pic_list) == 0:
            print("*********")
        else:
            for brand_img in brand_pic_list:
                # print(brand_img)
                signatureB_this, md_colorB_this = get_signature(brand_img)
                # print(signatureB_this, md_colorB_this)
                sB_list.append(signatureB_this)
                mB_list.append(md_colorB_this)
                tB_list.append(brand)
                
    print("target: ", len(sB_list), len(mB_list))
    try:

        a = np.array(sB_list, dtype=object)
        np.save("dataset/signatureB_targetlist.npy", a)
        b = np.array(mB_list)
        np.save("dataset/md_colorB_targetlist.npy", b)
        c = np.array(tB_list)
        np.save("dataset/targetbrand_targetlist.npy", c)
    except:
        print("Cannot save")
        pass

def subfunc_runsample(csv_id):
    print("csv {} process {}".format(csv_id, os.getpid()))
    targetlist = "dataset/targetlist/merge277"
    mode = "benign"  ## must specify mode is for phish/benign
    emd_ = 0.6#[0.94]
    N = 5#[1, 3, 5]#, 10] ## top1/3/5/10

    ## cache features for targetlist screenshots
    print("------0-read targetlist----")
    # get_targetlist_feature(targetlist)
    print(f"read targetlist feature using {time.time() - starttime} seconds")
    
    signatureB_list = np.load("dataset/signatureB_targetlist.npy", allow_pickle=True)
    md_colorB_list = np.load("dataset/md_colorB_targetlist.npy", allow_pickle=True)
    tar_list = np.load("dataset/targetbrand_targetlist.npy", allow_pickle=True)
    assert len(signatureB_list) == len(md_colorB_list) ## assert singature list and md_color list must have the same length
    assert len(signatureB_list) == len(tar_list) ## each target brand get 1 feature vector
    
    print("------1-read testing sample----")
    signatureA_list = []
    md_colorA_list = []
    
    df = pd.read_csv("data_test/data_test.csv")
    
    for iidx, rrow in df.iterrows():
        signatureA_this, md_colorA_this = get_signature(rrow["scr_path"])
        signatureA_list.append(signatureA_this)
        md_colorA_list.append(md_colorA_this)
    print("target: ", len(signatureA_list), len(md_colorA_list))
    
    # signatureA_list = np.array(signatureA_list)
    # md_colorA_list = np.array(md_colorA_list)
    # tar_listA = np.array(tar_listA)
    
    assert len(md_colorA_list) == len(signatureA_list) ## assert singature list and md_color list must have the same length
    
    
    # output
    normal_csv = open("results/eval_emd.csv", "w")
    normal_csvwriter = csv.writer(normal_csv)
    normal_csvwriter.writerow(["scr_path", "true_brand", "pred_brand", "pred_score", "pred_target_list", "pred_score_list"])#, "pic_path"])
    print("------2-calculate----")
    
    for idx, row in df.iterrows():
        ## open the screenshot
        # img1url = row["filename"]
        pred_target = None # top1
        pred_score = 0 # top1
        pred_target_list = []
        pred_score_list = []
        
        # pred_phish = 0 # == False, benign
        # topi = 0
        count_emd = []
        signatureA = signatureA_list[idx]
        md_colorA = md_colorA_list[idx]

        for i in range(len(signatureB_list)):
            signatureB = signatureB_list[i]
            md_colorB = md_colorB_list[i]
            tar = tar_list[i]
            try:
                # print(md_colorA, md_colorB)
                # print(signatureA, signatureB)
                emd = get_feature(signatureA, signatureB, md_colorA, md_colorB)
                # if emd > emd_:  # Find all brands that exceeds similarity threshold
                count_emd.append(Emd(emd, tar))
                    
            except Exception as e:
                # print(signatureA, signatureB, md_colorA, md_colorB)
                # # print(e)
                # print("Cannot compare features for: ", tar, ' and ', img1url)
                pass
        
        # if prediction is non-empty
        # print("count_emd", len(count_emd))
        if len(count_emd) != 0:
            # pred_phish = 1
            
            ## sort according to similarity
            cmpfun = operator.attrgetter('emd', 'targetlist_name')
            count_emd.sort(key=cmpfun, reverse=True)
            target_list = count_emd[:N]
            # print([item.targetlist_name for item in count_emd])
            pred_target = target_list[0].targetlist_name
            pred_score = target_list[0].emd
            pred_target_list = [target.targetlist_name for target in target_list]
            pred_score_list = [target.emd for target in target_list]

        else:
            pred_target = None
            pred_score = 0
            pred_target_list = []
            pred_score_list = []
        
        normal_csvwriter.writerow([row["scr_path"], row["brand"], pred_target, str(pred_score), str(pred_target_list), str(pred_score_list)])



if __name__ == '__main__':
    starttime = time.time()
    print(f"start time {starttime}")
    date = datetime.today().strftime('%Y-%m-%d')
    subfunc_runsample(0)
    print('All subprocesses done.')
    
    print("use time: {}".format(time.time() - starttime))
    print('Process finish')
