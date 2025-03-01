import time
import numpy as np
import argparse
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from datetime import datetime

from dataloader import APWGFeature, TargetFeature
from model import RedNet, BottleneckBlock
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# paddle.device.set_device('gpu:1')

class FeatureExtractor(nn.Layer):
    def __init__(self, pretrained_model):
        super(FeatureExtractor, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.fc = nn.Identity()
    def forward(self, x):
        x = self.pretrained_model(x)
        return x


def calculate_cos(test_path, target_path, map2classname):
    test_data = np.load(test_path)
    target_data = np.load(target_path)

    # 计算余弦相似度
    similarity = cosine_similarity(test_data, target_data)
    
    max_values = np.max(similarity, axis=1)
    max_index = np.argmax(similarity, axis=1)
    
    vfunc = np.vectorize(lambda x: map2classname.get(x, None))
    max_class = vfunc(max_index)
    return max_index, max_values, max_class

def get_targetlist_feature(args, tag277):
    model = RedNet(BottleneckBlock, 26)
    if tag277 == "expand277":
        # expand277
        test_custom_dataset = TargetFeature('dataset/expand277.csv', transforms, "/".join(args.targetlist.split("/")[:-1]))
        
    elif tag277 == "expand277_new":
        # expand277
        test_custom_dataset = TargetFeature('dataset/expand277_new.csv', transforms, "/".join(args.targetlist.split("/")[:-1]))
    
    params = paddle.load(args.weights)
    model.set_dict(params)
    test_loader = paddle.io.DataLoader(test_custom_dataset, batch_size=64, shuffle=False, num_workers=1, drop_last=False)
    
        
    feature_extractor = FeatureExtractor(model)

    feature_extractor.eval()
    # 存储特征
    features_list = []
    for images in test_loader:
        with paddle.no_grad():
            features = feature_extractor(images)
        features_list.append(features.numpy())

    # 合并特征
    features_array = np.concatenate(features_list, axis=0)

    # 保存特征到文件
    np.save(f"dataset/targetlist_{tag277}_feature.npy", features_array)
    print("finish")

def get_making_feature(args, crop_path, tag277):
    model = RedNet(BottleneckBlock, 26)
    test_custom_dataset = APWGFeature(crop_path, transforms)
    params = paddle.load(args.weights)
    model.set_dict(params)
    test_loader = paddle.io.DataLoader(test_custom_dataset, batch_size=64, shuffle=False, num_workers=1, drop_last=False)
    
        
    feature_extractor = FeatureExtractor(model)

    feature_extractor.eval()
    # 存储特征
    features_list = []
    for images in test_loader:
        with paddle.no_grad():
            features = feature_extractor(images)
        features_list.append(features.numpy())

    # 合并特征
    features_array = np.concatenate(features_list, axis=0)

    # 保存特征到文件
    np.save("dataset/testing_" + tag277 + "_feature.npy", features_array)
    print("finish")


if __name__ == '__main__':
    # parameters
    date = datetime.today().strftime('%Y-%m-%d')
    print('Today is:', date)
    start_time = datetime.now()
    startTime = time.time()
    print(f"Start Eval Time: {startTime}")
    
    device = "cpu"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_csv",
                        default="data_test/data_test.csv",
                        help='Input csv path to test')
    
    parser.add_argument("-output_csv", 
                        default="result_{}".format(date),
                        help='Output results file name')
    # weights parameter
    parser.add_argument('-weights',
                        type=str, 
                        required=True,
                        choices=["finetune277_models/final.pdparams", "finetune277_new_models/final.pdparams"],
                        help='model weights path')
    parser.add_argument('-targetlist',
                        type=str, 
                        required=True,
                        choices=["../../../data/targetlist/expand277", "../../../data/targetlist/expand277_new"],
                        help='Targetlist folder path')
    args = parser.parse_args()
    
    
    
    '''0. read dataset'''
    starttime = time.time()
    print(f"start: {starttime}")
    # read data
    transforms = T.Compose([
        T.Resize((224, 224)),
        # T.CenterCrop(224),
        T.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            data_format='HWC'),
        T.ToTensor()
    ])
    # read crop logo
    df_scr = pd.read_csv(args.input_csv)
    df_scr["logo_path"] = df_scr["scr_path"].replace(".png", "_crop_logo.png")
    '''get targeltlist feature'''
    tag277 = args.targetlist.split("/")[-1]
    get_targetlist_feature(args, tag277)   
    '''get testing sample feature'''
    get_making_feature(args, df_scr, tag277) # need to input the testing csv of cropped logo paths
    '''Calculate simialrity'''
    df_target = pd.read_csv(f"dataset/{tag277}.csv")
    target277_new_path = f"dataset/targetlist_{tag277}_feature.npy"
    testing_feature = "dataset/testing_" + tag277 + "_feature.npy"

    # targetlist 277 new
    df_target['indices'] = range(len(df_target))
    map_dict = dict(zip(df_target['indices'], df_target['brand']))
    
    pred_index, pred_value, pred_class = calculate_cos(testing_feature, target277_new_path, map_dict)
    # true brand 这里需要
    df_data = df_scr
    crop_logos = df_data["logo_path"].tolist()
    true_brands = df_data["brand"].tolist()
    df = pd.DataFrame({"logo_path":crop_logos, "pred_value":pred_value, "true_brand": true_brands, "pred_brand":pred_class})
    df.to_csv("results/eval_" + tag277+"_box_logo_similarity.csv", index=False)
    print("finish", tag277)
    print(f"end: {time.time()}, during: {time.time()-starttime}")