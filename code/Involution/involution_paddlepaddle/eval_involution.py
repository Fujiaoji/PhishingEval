import time
import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from dataloader import APWGFeature, TargetFeature
from model import RedNet, BottleneckBlock
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

paddle.device.set_device('gpu:1')

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

def get_targetlist_feature(tag277):
    model = RedNet(BottleneckBlock, 26)
    if tag277 == "expand277":
        # expand277
        test_custom_dataset = TargetFeature('dataset/expand277.csv', transforms)
        params = paddle.load("finetune277_models/final.pdparams")
        model.set_dict(params)
    elif tag277 == "expand277_new":
        # expand277
        test_custom_dataset = TargetFeature('dataset/expand277_new.csv', transforms)
        params = paddle.load("finetune277_new_models/final.pdparams")
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

def get_making_feature(crop_path, tag277):
    model = RedNet(BottleneckBlock, 26)
    test_custom_dataset = APWGFeature(crop_path, transforms)
    if tag277 == "expand277":
        params = paddle.load("finetune277_models/final.pdparams")
    else:
        params = paddle.load("finetune277_new_models/final.pdparams")
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
    '''get targeltlist feature'''
    tag277 = "expand277" # or "expand277_new"
    get_targetlist_feature(tag277)   
    '''get testing sample feature'''
    get_making_feature(pd.read_csv("data_test/data_test.csv"), tag277) # need to input the testing csv of cropped logo paths
    '''Calculate simialrity'''
    df_target = pd.read_csv(f"dataset/{tag277}.csv")
    target277_new_path = f"dataset/targetlist_{tag277}_feature.npy"
    testing_feature = "dataset/testing_" + tag277 + "_feature.npy"

    # targetlist 277 new
    df_target['indices'] = range(len(df_target))
    map_dict = dict(zip(df_target['indices'], df_target['brand']))
    
    pred_index, pred_value, pred_class = calculate_cos(testing_feature, target277_new_path, map_dict)
    # true brand 这里需要
    df_data = pd.read_csv("data_test/data_test.csv")
    crop_logos = df_data["logo_path"].tolist()
    true_brands = df_data["brand"].tolist()
    df = pd.DataFrame({"logo_path":crop_logos, "pred_value":pred_value, "true_brand": true_brands, "pred_brand":pred_class})
    df.to_csv("results/eval_" + tag277+"_box_logo_similarity.csv", index=False)
    print("finish", tag277)
    print(f"end: {time.time()}, during: {time.time()-starttime}")