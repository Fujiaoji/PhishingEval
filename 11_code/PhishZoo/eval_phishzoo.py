import os
import re
import csv
import time
import argparse

import pickle
import pandas as pd
import numpy as np
import cv2 as cv
import json
import multiprocessing

from multiprocessing import Pool
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils import brand_converter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class TokenSite:
    """Tokenize html and url"""

    def __init__(self, html, url):  ##input the html and the url
        self.html = html
        self.url = url

    def token_keyword_html(self):
        if len(self.html) == 0:
            return ""
        text = self.html

        # kill all script and style elements
        for script in text(["script", "style"]):
            script.extract()  # rip it out

        # get text
        text = text.get_text()
        text = re.sub(
            r'[`\=@©#$%^*()_+\[\]{};\'\\:"|<,./<>?‘’-]', " ", "".join(text.splitlines())
        )
        text = text.replace("\xa0", " ")
        text = text.split()
        for txt in text:
            if len(txt) < 3:
                text.remove(txt)
        if len(text) == 0:
            return ""
        else:
            return " ".join(text)

    def token_keyword_url(self):
        token_list = re.split(r'[`\=@©#$%^*()_+\[\]{};\'\\:"|<,./<>?‘’-]', self.url)
        for token in token_list:
            if ("www" in token) or ("https" in token) or ("http" in token) or ("com" in token) or token == "":
                token_list.remove(token)
        if len(token_list) == 0:
            return ""
        else:
            return " ".join(token_list)

    def output(self):
        html_tok = self.token_keyword_html()
        url_tok = self.token_keyword_url()
        final_tok = html_tok
        final_tok = final_tok + url_tok
        return final_tok


def dict_construct(target_root, domain_map_path):
    """dictionary constructor"""
    with open(domain_map_path, "rb") as handle:
        domain_map = pickle.load(handle)
    # print(domain_map)

    sample_dir_list = []
    dir_list = os.listdir(target_root)
    dir_list = [
        item
        for item in dir_list
        if (not item.startswith(".")) and (not item.endswith((".txt", ".npy", ".pkl")))
    ]

    for folder in dir_list:
        sample_dir = os.listdir(os.path.join(target_root, folder))
        sample_dir = [item for item in sample_dir if "html.txt" in item]
        if len(sample_dir) == 0:
            sample_dir_list.append(None)
        else:
            try:
                if "homepage_html.txt" in sample_dir:
                    sample_dir_list.append(os.path.join(target_root, folder, "homepage_html.txt"))
                else:
                    sample_dir_list.append(os.path.join(target_root, folder, "login_html.txt"))
            except:
                sample_dir_list.append(None)
    
    ground_brand = dir_list
    ground_domain = [domain_map[brand_converter(x)][0] for x in ground_brand]
    print("ground_domain", len(ground_domain))
    ground_html = []
    # extract html content
    for i in tqdm(range(len(sample_dir_list))):
        try:
            with open(sample_dir_list[i]) as handle:
                soup = BeautifulSoup(handle.read(), "html5lib")
                if len(soup) == 0:
                    ground_html.append("")
                    # print(f"--{sample_dir_list[i]} soup = 0--")
                else:
                    try:
                        ground_html.append(soup)
                    except:
                        ground_html.append("")
                        # print(f"--{sample_dir_list[i]} open error--")
        except:
            ground_html.append("")
            # print(f"--{sample_dir_list[i]} who knows--")

    # for idx, ited in enumerate(ground_html):
    #     if len(ited) == 0:
    #         print(ited)
    #     if (ited == "") or (ited == None):
    #         print(idx, sample_dir_list[idx])
    print("Reading completed...")
    assert len(ground_html) == len(ground_domain)
    print(len(ground_html))

    token_list = []
    for i in tqdm(range(len(ground_domain))):
        if len(ground_html[i]) != 0:
            token_this = TokenSite(ground_html[i], ground_domain[i])
            token_list.append(token_this.output())
        else:
            token_list.append(ground_domain[i])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(token_list)

    for i in range(X.shape[0]):
        # get the first vector out (for the first document)
        first_vector_tfidfvectorizer = X[i]
        # place tf-idf values in a pandas data frame, select first 5 most frequent terms
        df = pd.DataFrame(
            first_vector_tfidfvectorizer.T.todense(),
            index=vectorizer.get_feature_names(),
            columns=["tfidf"],
        )
        df = df.sort_values(by=["tfidf"], ascending=False).iloc[:5, :]
        # print(df)
        df.to_csv(target_root + "/" + ground_brand[i] + "/tfidf.csv")
        # break


def getMatchNum(matches, ratio):
    """return number of matched keypoints"""
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # compute good matches
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)


class SIFT:
    """SIFT extractor"""

    def __init__(self, login_path, brand_folder):
        self.login_path = login_path
        self.brand_folder = brand_folder
        self.logo_kp_list, self.logo_des_list, self.logo_file_list = self.logo_kp()
        assert len(self.logo_kp_list) > 0
        assert len(self.logo_kp_list) == len(self.logo_des_list)
        assert len(self.logo_file_list) == len(self.logo_des_list)

    def match(self):
        # construct sift extractor
        sift = cv.xfeatures2d.SIFT_create()
        # FLANN match
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv.FlannBasedMatcher(indexParams, searchParams)

        # extract webpage kp
        try:
            queryImage = cv.imread(self.login_path, cv.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(queryImage, None)
        except Exception as e:
            print(e)
            print("Cannot identify the screenshot")
            return self.brand_folder.split("/")[-1], None, 0

        similarity = []
        filename = []
        for i in range(len(self.logo_kp_list)):
            # extract kp from logo
            logo_kp = self.logo_kp_list[i]
            logo_des = self.logo_des_list[i]
            try:
                matches = flann.knnMatch(logo_des, des, k=2)  # match keypoint
                (matchNum, matchesMask) = getMatchNum(
                    matches, 0.9
                )  # calculate matchratio
                matchRatio = matchNum * 100 / len(logo_kp)
            except:
                matchRatio = 0
            similarity.append(matchRatio)
            filename.append(self.logo_file_list[i])
            del matchRatio  ## delete matchRatio and go to next round

        maxfilename = np.array(filename)[np.argsort(similarity)[::-1]][
            0
        ]  ## which logo gives the maximum similarity
        maxscore = max(similarity)  ## maximum similarity
        return self.brand_folder.split("/")[-1], maxfilename, maxscore

    def logo_kp(self):
        """target list"""
        sift = cv.xfeatures2d.SIFT_create()
        img_kp_list = []
        img_des_list = []
        img_file_list = []
        for file in os.listdir(self.brand_folder):
            if (
                not file.startswith("loginpage")
                and not file.startswith("homepage")
                and file.endswith(("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"))
            ):
                try:
                    img = cv.imread(
                        self.brand_folder + "/" + file, cv.IMREAD_GRAYSCALE
                    )  ## convert to grayscale
                    kp, des = sift.detectAndCompute(img, None)
                    img_kp_list.append(kp)
                    img_des_list.append(des)
                    img_file_list.append(self.brand_folder + "/" + file)
                except Exception as e:
                    print(e)
        return img_kp_list, img_des_list, img_file_list


def get_content(html_path):
    content = ""
    try:
        with open(html_path) as handle:
            soup = BeautifulSoup(handle.read(), "html5lib")
            if len(soup) == 0:
                content = ""
            else:
                try:
                    content = soup
                except:
                    content = ""
    except:
        content = ""
    return content


def run_subfuc_eval(args):
    print("child process {}".format(os.getpid()))
    # log file
    flog = open(args.log, "w")
    print("child process {}".format(os.getpid()), file=flog)
    starttime = time.time()
    print("start time", starttime, file=flog)
    
    # dict_construct(args.targetlist, args.domain_path)
    ts = 40

    # # load targetlist and model
    print("read csv", file=flog)
    # replace the csv you wnat to test
    df = pd.read_csv(args.input_csv)
    normal_csv = open(args.result_path, "w")
    normal_csvwriter = csv.writer(normal_csv)
    normal_csvwriter.writerow(["scr_path", "phish", "sim", "true_brand", "pred_brand", "top5_sim", "top5_target"])
    
    # read the target list info
    print("read target list", file=flog)
    ground_token_dict = {}

    folders = os.listdir(args.targetlist)
    folders = [item for item in folders if (not item.startswith(".")) and (not item.endswith((".txt", ".npy", ".pkl")))]
    for folder in folders:
        try:
            ground_token = pd.read_csv(args.targetlist + "/" + folder + "/tfidf.csv")
        except:
            continue
        ground_token = list(ground_token.iloc[:, 0])  ## get first column
        ground_token = [cc.lower() for cc in ground_token]
        ground_token_dict[folder] = ground_token

    # for the testing dataset
    print("testing", file=flog)
    for index, row in df.iterrows():
        html_path = row["scr_path"].replace(".png", ".html")
        html_content = get_content(html_path)

        """Make SIFT prediction only if it contains popular(Top 5) tokens from targeted brand"""
        # tokenize this site (html + url)
        url = row["domain"]
        check = TokenSite(html_content, url)
        web_token = check.output()
        web_token_list = [word.lower() for word in web_token.split()]
        print(index, web_token_list, file=flog)
        ## check against protected logos
        pred_brand = []
        similarity = []
        """this part is time consuming, polish this part"""
        for key, item in ground_token_dict.items():
            if len(item) > 0:
                if len(set(item) & set(web_token_list)) > 0:
                    check = SIFT(row["scr_path"],args.targetlist + "/" + key)
                    _, _, simscore = check.match()
                    similarity.append(simscore)
                    pred_brand.append(key)
                    # break

        assert len(similarity) == len(
            pred_brand
        )  ## each protected brand is associated with 1 similarity

        ## sort according to similarity in descending order
        pred_brand_sort = np.array(pred_brand)[np.argsort(similarity)[::-1]]
        similarity_sort = np.array(similarity)[np.argsort(similarity)[::-1]]

        ## filter predictions which exceed threshold t_s
        pred_brand_sort_filter = pred_brand_sort[similarity_sort > ts]
        similarity_sort_filter = similarity_sort[similarity_sort > ts]
        assert len(pred_brand_sort_filter) == len(similarity_sort_filter)

        pred_target = None
        sim = 0
        phish = 0
        ## if prediction is not None, pred_target = top1
        if len(similarity_sort_filter) > 0:
            phish = 1
            pred_target = pred_brand_sort_filter[0]
            sim = similarity_sort_filter[0]
        normal_csvwriter.writerow(
            [
                row["scr_path"],
                str(phish),
                str(sim),
                str(row["brand"]),
                pred_target,
                str(similarity_sort_filter[:5]),
                str(pred_brand_sort_filter[:5]),
            ]
        )
        
    print(
        "Finish time: {}".format(time.time() - starttime), file=flog
    )
    flog.close()


if __name__ == "__main__":
    starttime = time.time()
    parser = argparse.ArgumentParser(description='parameters')

    parser.add_argument('--log', type=str, default='results/result_log.txt', help='log file path')
    parser.add_argument('--targetlist', type=str, default='dataset/expand277', help='targetlist path') # replace with targetlist path
    parser.add_argument('--domain-path', type=str, default='domain_map.pkl', help='domain path')
    parser.add_argument('--input-csv', type=str, default='data_test/data_test.csv', help='result path')
    parser.add_argument('--result-path', type=str, default='results/result_eval.csv', help='input csv path')
    args = parser.parse_args()

    run_subfuc_eval(args)

    print("use time: {}".format(time.time() - starttime))
    print("Process finish")
