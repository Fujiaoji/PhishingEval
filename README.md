# Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Code](#code)
  - [reproduce_phishpedia](#reproduce_phishpedia)
  - [reproduce_phishintention](#reproduce_phishintention)
  - [run_DynaPhish](#run_DynaPhish)
  - [Involution](#Involution)
  - [VisualPhishNet](#VisualPhishNet)
  - [PhishZoo](#PhishZoo)
  - [EMD](#EMD)
- [Data](#data)
  - [targetlist](#targetlist)
  - [apwg451514](#apwg451514)
  - [archive100](#archive100)
  - [crawl_benign](#crawl_benign)
  - [failed_example_csv](#failed_example_csv)
  - [perturbated_dataset](#perturbated_dataset)
  - [visible_dataset2](#visible_dataset2)

# Introduction
The README.md is still updating. Check our website to access the newest version!

This is the official implementation of "Evaluating the Effectiveness and Robustness of Visual Similarity-based Phishing Detection Models" USENIX'25. Due to the space limitation, the full version of the paper is available at [link to arxiv](https://arxiv.org/abs/2405.19598), website is at [PhishingEval Website](https://moa-lab.net/evaluation-visual-similarity-based-phishing-detection-models/). Please based on the github to set the model structures since we divide the fodler into different parts in Zenodo.
Original codes for different methods are updated quickly, you can refer to their original code repos to access the newest codes.
# Models
- Due to its large size, we share the trained models on [OneDrive](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EiDwgElIisBAjA7H5LUnAL0BZQFdvtbTjXR_c03MWsKkgw?e=6CC3dm)
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
│── targetlist: target lists
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
```
### Preparation
1. **Download needed files**. Before running the code, please manually download targetlist to the ```PhishingEval/data/targetlist```. Then, please download the model weights through ```bash download_model.sh``` or manually download through the shared links. The model weights will be saved to ```reproduce_phishpedia/models```.
2. **Environment**.
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/install/)
- Create the env based on ```env_phishpedia.yml``` by ```conda env create -f env_phishpedia.yml```
- ```conda activate env_phishpedia```
- There are two more env need install ```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```, then ```pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"```

- Sample to run the code: ```conda activate env_phishpedia``` -> ```python eval_phishpedia.py -siamese_weights=<siamese bit model file path, eg models/bit_new.pth.tar> -targetlist=<targetlist folder path, eg. ../../data/targetlist/expand277_new>```
- Citation
```bibtex
@inproceedings{lin2021phishpedia,
  title={Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages},
  author={Lin, Yun and Liu, Ruofan and Divakaran, Dinil Mon and Ng, Jun Yang and Chan, Qing Zhou and Lu, Yiwen and Si, Yuxuan and Zhang, Fan and Dong, Jin Song},
  booktitle={30th USENIX Security Symposium},
  year={2021}
}
```
## reproduce_phishintention
- Original code link [PhishIntention](https://github.com/lindsey98/PhishIntention)
- input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.
- conda env: please install conda env based on original github env, and then follow the version of env_phishintention.yml
- download the trained_models folder and put it under repreduce_phishintention.
- command to run the code: ```conda activate env_phishintention``` -> ```python eval_phishintention.py```
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
# Data
- Note: Due to its large size, we put the subset of sampled phishing 4190 dataset corresponding to Table 3 in the paper. The apwg451514 is shared through the other link.
- targetlist: reference lists
    - expand277: PhishIntention-based logo reference list
    - expand277_new: expanded logo-based logo reference list
    - merge277: screenshot-based logo reference list
    - merge277_new: expanded screenshot-based logo reference list
- apwg451514: contains html, screenshots from apwg. Due to its large size, we share it at [this link](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/Era-x5Nn5NJLrTMQC4gLmicBuru_eAWhLd-H96K50ppwnQ?e=S6dgbS)
- archive100: 100 domain (Tranco1000) with html and screenshots (archive.org)
- crawl_benign: benign 110 brands' data, including:
    - login.png: screenshot
    - login.txt: url
    - login.html: html
    - classes.txt: class for label
    - XXX-login.txt: logo region
- failed_example_csv: the csv contains the html and screenshot paths that let models fail. Extract screenshot and html from apwg451514
- perturbated_dataset: black box attack and white box attach
- visible_dataset2: visible manipulation produced images
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

