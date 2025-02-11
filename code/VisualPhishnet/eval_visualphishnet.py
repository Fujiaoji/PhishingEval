import os
import csv
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten,Subtract,Reshape
# from keras.preprocessing import image
# from keras.models import Model
# from keras.regularizers import l2
from matplotlib.pyplot import imread
from keras import optimizers

from visualphish_model import *

# from multiprocessing import Pool
# import multiprocessing

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_weight_model(model_path):
    '''
    Load model
    :param model_path:
    :return: final_model: complete model which returns class
    :return: inside_model: partial model which returns an intermediate embedding
    '''
    margin = 2.2
    input_shape = [224, 224, 3]
    new_conv_params = [5, 5, 512]

    final_model = define_triplet_network(input_shape, new_conv_params)
    final_model.summary()
    optimizer = optimizers.Adam(lr = 0.0002)
    final_model.compile(loss=custom_loss(margin),optimizer=optimizer)
    final_model.load_weights(model_path)
    inside_model = final_model.layers[3]  # partial model to get the embedding
    return final_model, inside_model

# load targetlist embedding
def load_targetemb(emb_path, label_path, file_name_path):
    '''
    load targetlist embedding
    :return:
    '''
    targetlist_emb = np.load(emb_path)
    all_labels = np.load(label_path)
    all_file_names = np.load(file_name_path)
    return targetlist_emb, all_labels, all_file_names


# Find Smallest n distances
def find_min_distances(distances, n):
    idx = distances.argsort()[:n]
    values = distances[idx]
    return idx, values


# Find names of examples with min distance
def find_names_min_distances(idx, values, all_file_names):
    names_min_distance = ''
    only_names = []
    distances = ''
    for i in range(idx.shape[0]):
        index_min_distance = idx[i]
        names_min_distance = names_min_distance + 'Targetlist: ' + all_file_names[index_min_distance] + ','
        only_names.append(all_file_names[index_min_distance])
        distances = distances + str(values[i]) + ','

    names_min_distance = names_min_distance[:-1]
    distances = distances[:-1]

    return names_min_distance, only_names, distances

def read_data(csv_path, des_folder):
    '''
    read data
    :param data_path:
    :param reshape_size:
    :param chunk_range: Tuple
    :return:
    '''
    reshape_size=[224,224,3]
    print("read data {}".format(csv_path))
    rall_imgs = []
    rall_filename = []
    
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        print("-------", row["scr_path"])
        # if index% 1000 == 0:
        #     print(index)
        try:
            img = Image.open(row["scr_path"])
            if img.mode == 'P' or img.mode == "L":
                img = img.convert('RGB')
            img= np.array(img)
            img = img[:, :, 0:3] # RGB channel
            rall_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
            rall_filename.append(row["scr_path"])
        except:
            print(row["scr_path"], "img error")
            continue
    try:
        rall_imgs = np.asarray(rall_imgs)
        np.save(os.path.join(des_folder, "testing_embInitial"), rall_imgs)
        rall_filename = np.asarray(rall_filename)
        np.save(os.path.join(des_folder, "testing_filename"), rall_filename)
    except:
        print("save error")
    print("Finish read testing {}".format(csv_path))
    return rall_imgs, rall_filename

def read_targetlist_screenshot(targetlist_name, des_folder, reshape_size):
    rall_imgs = []
    rall_labels = []
    rall_file_names = []
    folder_path = os.path.join("../../data/targetlist", targetlist_name)
    folder_brands = os.listdir(folder_path)
    folder_brands = [item for item in folder_brands if (not item.startswith(".")) and (not item.endswith("pkl")) and (not item.endswith("txt"))]
    for idx, brand in enumerate(folder_brands[:2]):
        print("finish {}".format(idx))
        brand_img_list = os.listdir(os.path.join(folder_path, brand))
        if len(brand_img_list) > 0:
            brand_img_list = [item for item in brand_img_list if item.startswith("T")]#(item.endswith(".png"))]# and 
            for brand_img in brand_img_list:
                brand_img_path = os.path.join(folder_path, brand, brand_img)
                try:
                    img = imread(brand_img_path)
                    img = img[:,:,0:3]
                    rall_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
                    rall_labels.append(brand)
                    rall_file_names.append(brand_img_path)
              
                except:
                    #some images were saved with a wrong extensions 
                    try:
                        img = imread(brand_img_path,format='jpeg')
                        img = img[:,:,0:3]
                        rall_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
                        rall_labels.append(brand)
                        rall_file_names.append(brand_img_path)
                    except:
                        print('failed at:', brand_img_path)
                        
                        break 
                # try:
                #     img = Image.open(brand_img_path)
                #     if img.mode == 'P' or img.mode == "L":
                #         img = img.convert('RGB')
                #     img= np.array(img)
                #     img = img[:, :, 0:3] # RGB channels
                #     rall_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
                #     rall_labels.append(brand)
                #     rall_file_names.append(brand_img_path)
                # except:
                #     print(brand_img_path, "img error")
                #     continue
    rall_imgs = np.asarray(rall_imgs)
    rall_labels = np.asarray(rall_labels)
    rall_file_names = np.asarray(rall_file_names)
    np.save(os.path.join(des_folder, "targetlist_emb_initial.npy"), rall_imgs)
    np.save(os.path.join(des_folder, "targetlist_label.npy"), rall_labels)
    np.save(os.path.join(des_folder, "targetlist_filename.npy"), rall_file_names)


def func_eval(args):
    # parameters
    startt = time.time()
    ts = 1
    mode = "benign"
    
    reshape_size = [224,224,3]

    # folder: save results
    result_folder = "results"
    os.makedirs(result_folder, exist_ok=True)
    
    normal_csv = open(os.path.join(result_folder, args.output_csv), "w")
    normal_csvwriter = csv.writer(normal_csv)
    normal_csvwriter.writerow(["scr_path", "phish", "min_distances", "only_name", "Closest"])
    
    print("0-get target ini emb")
    read_targetlist_screenshot(args.targetlist, result_folder, reshape_size)
    img_initial_feature = np.load(os.path.join(result_folder, "targetlist_emb_initial.npy"))
    print("target img_initial_feature", img_initial_feature.shape)

    print("1-get model target embedding")
    _, inside_model = load_weight_model(args.model_path)
    target_data_emb = inside_model.predict(img_initial_feature, batch_size=32)
    # print("target_data_emb", target_data_emb.shape)
    np.save(os.path.join(result_folder, "targetlist_emb_model.npy"), target_data_emb)
    
    print(f"2-load target emb")
    targetlist_emb_path = os.path.join(result_folder, "targetlist_emb_model.npy")
    targetlist_label_path = os.path.join(result_folder, "targetlist_label.npy")
    targetlist_file_name_path = os.path.join(result_folder, "targetlist_filename.npy")
    targetlist_emb, all_labels, all_file_names = load_targetemb(targetlist_emb_path, targetlist_label_path, targetlist_file_name_path)
    print(targetlist_emb.shape, all_labels.shape, all_file_names.shape)
    # all_file_names = [x.split("/")[-1] for x in all_file_names]
    print('Loaded targetlist and model, number of protected target screenshots {}'.format(targetlist_emb.shape))
    
    # read data
    print("3-read testing data")
    X, _ = read_data(args.input_csv, result_folder)

    X_filename = pd.read_csv(args.input_csv)
    # X_filename = X_filename.head(2)
    print('Finish reading data, number of data {}'.format(len(X)))

    # get embeddings for testing data
    print("4-get testing embedding")
    test_data_emb = inside_model.predict(X, batch_size=32)
    print("test_data_emb", test_data_emb.shape)
    pairwise_distance = compute_all_distances(test_data_emb, targetlist_emb)
    
    print("pairwise_distance", pairwise_distance.shape)
    
    # # # 
    print('Finish getting embedding')
    n = 1 #Top-1 match
    print('Start ')

    for i, row in X_filename.iterrows():
        if i% 1000 == 0:
            print(i)
        phish = 0
        
        distances_to_target = pairwise_distance[i,:]
        
        idx, values = find_min_distances(np.ravel(distances_to_target), n)
        # print("idx, values", idx, values)
        
        names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)
        # distance lower than threshold ==> report as phishing
        if float(min_distances) <= ts:
            phish = 1
        # else it is benign
        else:
            phish = 0

        normal_csvwriter.writerow([row["scr_path"], str(phish), str(round(float(min_distances), 4)), str(only_names[0]), names_min_distance])

    print("finish task: {}".format((time.time() - startt)))



if __name__ == "__main__":
    date = datetime.today().strftime('%Y-%m-%d')
    starttime = time.time()
    print(f"start time: {starttime}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--input_csv",
                        default="data_test/data_test.csv",
                        help='Input dataset csv file')
    parser.add_argument('-r', 
                        "--output_csv", default="eval_result_{}.csv".format(date),
                        help='Output results csv')
    
    parser.add_argument('-t', "--targetlist", required=True, type=str,
                        help='merge277 or merge277_new')
    parser.add_argument('-m', "--model_path", 
                        default="models/model277_new_2.h5")
    args = parser.parse_args()

    func_eval(args)
    endtime = time.time()
    print(f"end time: {endtime}")
    print("use time: {}".format(endtime - starttime))
    