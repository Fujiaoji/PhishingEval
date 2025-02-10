# Table of Contents
- [Introduction](#introduction)
- [Data](#data)
  - [targetlist](#targetlist)
  - [apwg451514](#apwg451514)
  - [archive100](#archive100)
  - [crawl_benign](#crawl_benign)
  - [failed_example_csv](#failed_example_csv)
  - [perturbated_dataset](#perturbated_dataset)
  - [visible_dataset2](#visible_dataset2)
- [Code](#code)
  - [reproduce_phishpedia](#reproduce_phishpedia)
  - [reproduce_phishintention](#reproduce_phishintention)
  - [run_DynaPhish](#run_DynaPhish)
  - [Involution](#Involution)
  - [VisualPhishNet](#VisualPhishNet)
  - [PhishZoo](#PhishZoo)
  - [EMD](#EMD)

# Introduction
The README.md is still updating. Check our website to access the newest version!

This is the official implementation of "Evaluating the Effectiveness and Robustness of Visual Similarity-based Phishing Detection Models" USENIX'25. Due to the space limitation, the full version of the paper is available at [link to arxiv](https://arxiv.org/abs/2405.19598), website is at [PhishingEval Website](https://moa-lab.net/evaluation-visual-similarity-based-phishing-detection-models/), dataset can be obtained from our website or [Zenodo](https://zenodo.org/records/14668190)Please based on the github to set the model structures since we divide the fodler into different parts in Zenodo.
Original codes for different methods are updated quickly, you can refer to their original code repos to access the newest codes.

Note that, it is highly recommend to directly download the github repo through ZIP rather than git clone since there are lots of histories make it very slow. Download ZIP and rename it the PhishingEval.

# Data
- Please download the dataset under the ```PhishingEval/data``` forlder manually or by the ```bash download_data.sh <name of the dataset, e.g., expand277_new>```.
- Note: Due to its large size, we share the subset of sampled phishing 4190 dataset corresponding to Table 3 in the paper. The apwg451514 is shared through our website. Our datasets can be download through the their links.
- phishing4190: [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EjMqfQrhvj5Mon7Hopf7WE8BtmnR4Z67KgiD2DSWbN1hkg?e=HHdtr2)
- expand277: PhishIntention-based logo reference list [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/ErcF2zwlYIhDomPZV5jIuisBASFG8TZ_LTZVW2ASpXF2Jw?e=SWOJ1B)
- expand277_new: expanded logo-based logo reference list [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EsLbpk8hZcNCgoqxFqt5Q9oBXpJuLY9eUdNT6-vaMYdSPQ?e=gottTM)
- merge277: screenshot-based logo reference list [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EkaMHem2lipPtcykRvaoS6YBlB6dBWmAD0PwKqiL6hBQLg?e=ICZD9E)
- merge277_new: expanded screenshot-based logo reference list [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EogoITGNTyVPr2Cl6gsEw8wBo3YDF79Ir_HHqTF8NILVqg?e=jDwLt8)
- apwg451514: contains html, screenshots from apwg. Due to its large size, we share it at our website.
- archive100: 100 domain (Tranco1000) with html and screenshots (archive.org) [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EkTnJVQ2mRNJhgFsuR_bLecBixzI-MK-yqk4PnLNy43dwA?e=5ZeP94)
- crawl_benign: benign 110 brands' data, including: [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EqG6_TChz19CjZpksPf_c80BuY1DDBOeZZgvoVUNSIswPQ?e=VBKB9T)
    - login.png: screenshot
    - login.txt: url
    - login.html: html
    - classes.txt: class for label
    - XXX-login.txt: logo region
- failed_example_csv [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EpZhWqg-mYNEh1RINeylDLEBub3CiBdCuhYWEb3Bup06lA?e=bNbYiU): the csv contains the html and screenshot paths that let models fail. Extract screenshot and html from apwg451514
- perturbated_dataset [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EuORCLXB5D9LpWHrgKUpb7sBQ8NKdbEQ-6XJzX46dWdXkA?e=vIiHiv): black box attack and white box attach
- visible_dataset2 [Onedrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/El8Pl95cvglLtr0nUtPan3IB-p5EMsVj7pf-ribb4CkobA?e=vEhCm7): visible manipulation produced images
    - 00: Elimination
    - 01: Color Replacement
    - 02: Scaling
    - 03: Rotation
    - 04: Integration
    - 05: Location
    - 06: Flipping
    - 07: Replacement
    - 08: Blurring
    - 09: Resizing
    - 10: Omission
    - 11: Case
    - 12: Font
    - Fonts: used for changing fonts
    - LogoLabelStudio: cropped logo, textual part logo, and image part logo 
# Code
## reproduce_phishpedia
Original code repository is at [Phishpedia](https://github.com/lindsey98/Phishpedia).
### Structure
```
reproduce_phishpedia/
│── configs
│── models: trained models
│   ├── bit.pth.tar: baseline weights, used with targetlist/expand277
│   ├── bit_new.pth.tar: extended weights, used with targetlist/expand277_new
│   ├── model_final.pth: ele weights
│   ├── domain_map.pkl: save the brand-domain information
│── results
│── train_ob
│   ├── inference_ob.py
│── train_siamese
│   ├── inference_siamese.py
│   ├── utils.py
│── env_phishpeida.yml
│── phishpedia_config.py
│── siamese.py
│── models.py
│── eval_phishpedia.py # evaluation file
│── download_model.sh # bash file to download trained models
```
### Preparation
0. **Download the repo**. Download the repo and rename it to ```PhishingEval```.
1. **Download needed files**. Before running the code, please manually download targetlist to the ```PhishingEval/data/targetlist```. Then, please download the model weights through ```cd PhishingEval/code/reproduce_phishpedia``` -> ```bash download_model.sh``` or manually download through the shared links. The model weights will be saved to ```reproduce_phishpedia/models```.
2. **Environment**.
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/install/)
- Create the env based on ```env_phishpedia.yml``` by ```conda env create -f env_phishpedia.yml```
- ```conda activate env_phishpedia```
- There are two more env need install ```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```, then ```pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"```

- Run the code for the sample: ```conda activate env_phishpedia``` -> ```python eval_phishpedia.py -siamese_weights=<siamese bit model file path, eg models/bit_new.pth.tar> -targetlist=<targetlist folder path, eg. "../../data/targetlist/expand277_new>"```
3. **Prepare Input**. The input should be similar style with ```data_test.csv```.

### Citation
```bibtex
@inproceedings{lin2021phishpedia,
  title={Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages},
  author={Lin, Yun and Liu, Ruofan and Divakaran, Dinil Mon and Ng, Jun Yang and Chan, Qing Zhou and Lu, Yiwen and Si, Yuxuan and Zhang, Fan and Dong, Jin Song},
  booktitle={30th USENIX Security Symposium},
  year={2021}
}
```
## reproduce_phishintention
Original code link [PhishIntention](https://github.com/lindsey98/PhishIntention)
### Structure
```
reproduce_phishintention/
│── AWL
│── CRP_Classifier
│── OCR_Siamese
│── data_test
│── results
│── env_phishintention.yml
│── phishintention_config.py
│── eval_phishintention.py # evaluation file
│── download_model.sh # bash file to download trained models
```
### Preparation
1. **Environment** 
- Same as phjishpedia, make sure you have download the target list and the data you want to test
- ```bash download_model.sh``` to download the trained models
- Create the env based on ```env_phishintention.yml``` by ```conda env create -f env_phishintention.yml```
- ```conda activate env_phishintention```
- There are two more env need install ```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```, then ```pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"```. Then ```pip install webdriver-manager==4.0.2```.

- command to run the code: ```conda activate env_phishintention``` -> ```python eval_phishintention.py --expand="N"```.

2. **Prepare Input**. Input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.

- Citation
```bibtex
@inproceedings{liu2022inferring,
  title={Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision Based Approach},
  author={Liu, Ruofan and Lin, Yun and Yang, Xianglin and Ng, Siang Hwee and Divakaran, Dinil Mon and Dong, Jin Song},
  booktitle={30th USENIX Security Symposium Security},
  year={2022}
}
```
## run_DynaPhish
- Original code link [DynaPhish](https://github.com/code-philia/Dynaphish)
- input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.
- trained_models: same as the reproduce_phishintention, need to change to the same path after isntalling the conda env
- conda env: please install conda env based on original github env, and then follow the version of rundy.yml
- command: ```conda activate rundy``` ->```python -m field_study_logo2brand.dynaphish_main```
- Citation
```bibtex
@inproceedings {291106,
    title = {Knowledge Expansion and Counterfactual Interaction for {Reference-Based} Phishing Detection},
    author = {Ruofan Liu and Yun Lin and Yifan Zhang and Penn Han Lee and Jin Song Dong},
    booktitle = {32nd USENIX Security Symposium},
    year = {2023}
}
```
## Involution
- Original code link [Involution](https://github.com/d-li14/involution)
- conda env: env_involution.yml
- eval:
  - first use Phishpedia top1 box to crop the logo
  - extract the target list info to csv
  - then eval by ```python eval_involution.py```
- Citation
```bibtex
@InProceedings{Li_2021_CVPR,
    author = {Li, Duo and Hu, Jie and Wang, Changhu and Li, Xiangtai and She, Qi and Zhu, Lei and Zhang, Tong and Chen, Qifeng},
    title = {Involution: Inverting the Inherence of Convolution for Visual Recognition},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```
## VisualPhishNet
- Original code link [VisualPhishNet](https://github.com/S-Abdelnabi/VisualPhishNet), other reference code link[PhishBaseline](https://github.com/lindsey98/PhishingBaseline)
- Input: screenshot
- conda env: env_visualphishnet.yml
- command: ```python eval_visualphishnet.py```
- Citation
```bibtex
@inproceedings{abdelnabi20ccs,
title = {VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity},
author = {Sahar Abdelnabi and Katharina Krombholz and Mario Fritz},
year = {2020},
booktitle = {ACM Conference on Computer and Communications Security (CCS) }
}
```
## PhishZoo
- Reference code link [PhishZoo](https://github.com/lindsey98/PhishingBaseline)
- input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.
- conda env: env_phishzoo.yml
- command to run the code: ```conda activate env_phishzoo``` -> ```python eval_phishzoo.py```
## EMD
- Reference code link [EMD](https://github.com/lindsey98/PhishingBaseline)
- input data information: screenshot
- can use the former env to run the code: ```python eval_emd.py```
