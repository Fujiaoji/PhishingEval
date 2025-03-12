# Table of Contents
- [Introduction](#introduction)
  - [total_structure](#total-structure)
- [Data](#data)
  - [targetlist](#targetlist)
  - [apwg451514](#apwg451514)
  - [phishing4190](#phishing4190)
  - [failed_example_csv](#failed_example_csv)
  - [archive100](#archive100)
  - [crawl_benign](#crawl_benign)
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
The README.md is still being updated. Please check our github to access the newest version!

This is the official implementation of "Evaluating the Effectiveness and Robustness of Visual Similarity-based Phishing Detection Models" USENIX'25. Due to the space limitation, the full version of the paper is available at [link to arxiv](https://arxiv.org/abs/2405.19598). Additional resources, including the website and dataset access, can be found at:
- Website: [PhishingEval Website](https://moa-lab.net/evaluation-visual-similarity-based-phishing-detection-models/)
- Dataset: Available on our website or via [Zenodo](https://zenodo.org/records/14668190). 

Please based on the github to set the model structures since we divide the folders into different parts in Zenodo.
Original codes for different methods are updated quickly, you can refer to their original code repos to access the newest codes.

The original implementations of different methods are updated frequently. To access the latest versions, please check their respective repositories.

Note: We recommend **directly downloading the github repository as a ZIP file** rather than git clone since there are lots of histories make it very slow. Download ZIP and rename it the PhishingEval.
## Total Structure
```
PhishingEval/
│── data
│   ├──data_test: used for testing 3 samples
│   ├──targetlist
│   ├──│──expand277
│   ├──│──expand277_new
│   ├──│──merge277
│   ├──│──merge277_new
│   ├──apwg451514
│   ├──phishing4190: a small dataset for testing different models
│   ├──failed_example_csv
│   ├──archive100
│   ├──crawl_benign
│   ├──perturbated_dataset
│   ├──visible_dataset2
│── code
│   ├──analyze: used for analyzing results
│   ├──reproduce_phishpedia
│   ├──reproduce_phishintention
│   ├──run_DynaPhish
│   ├──Involution
│   ├──PhishZoo
│   ├──VisualPhishNet
│   ├──EMD
│── download_data.sh: used for download dataset
│── LICENSE
│── README.md
```

# Data
Please download the dataset under the `PhishingEval/data` folder.
## targetlist
- expand277: PhishIntention-based logo reference list 
- expand277_new: expanded logo-based logo reference list 
- merge277: screenshot-based logo reference list 
- merge277_new: expanded screenshot-based logo reference list
## apwg451514
contains html, screenshots from apwg. The apwg451514 is shared through our website. Our datasets can be download through the their links.
## phishing4190
Due to the large size of apwg451514, we share the subset of sampled phishing 4190 dataset corresponding to Table 3 in the paper.
## failed_example_csv: the csv contains the html and screenshot paths that let models fail. Extract screenshot and html from apwg451514
## archive100
100 domain (Tranco1000) with html and screenshots (archive.org) 
## crawl_benign
benign 110 brands' data, including: 
    - login.png: screenshot
    - login.txt: url
    - login.html: html
    - classes.txt: class for label
    - XXX-login.txt: logo region
## perturbated_dataset: black box attack and white box attach
## visible_dataset2: visible manipulation produced images:
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
- **Download the repository**. Directly download the GitHub repo Zip, unzip, and rename it to ```PhishingEval```
- **Run Testing Sample**. `cd PhishingEval`, `bash download_data.sh expand277` and `bash download_data.sh merge277` to download the targetlist.
- **Run Table 3**. `cd PhishingEval`, `bash download_data.sh phishing4190` to download the testing dataset. `bash download_data.sh expand277` and `bash download_data.sh merge277` to download the targetlist.
- Note: merge277: 11.4G, expand277: 243M

## reproduce_phishpedia
Original code repository is at [Phishpedia](https://github.com/lindsey98/Phishpedia).
### Structure
```
reproduce_phishpedia/
│── configs/
│── models/: trained models
│   ├── bit.pth.tar: baseline weights, used with targetlist/expand277
│   ├── bit_new.pth.tar: extended weights, used with targetlist/expand277_new
│   ├── model_final.pth: ele weights
│   ├── domain_map.pkl: save the brand-domain information
│── train_ob/
│   ├── inference_ob.py
│── train_siamese/
│   ├── inference_siamese.py
│   ├── utils.py
│── phishpedia_config.py
│── siamese.py
│── models.py
│── eval_phishpedia.py # evaluation file
│── download_model.sh # bash file to download trained models
│── env_phishpeida.yml
│── requirement.txt
│── setup_cpu.sh
```
### Preparation
1. **Download Model Weights**. 
  - `cd PhishingEval/code/reproduce_phishpedia`
  - `bash download_model.sh`. The model weights will be saved to `reproduce_phishpedia/models`.
2. **Environment**
  - Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/install/)
  - **CPU Version Install** 
    - `bash setup_cpu.sh`
    - `conda activate env_phishpedia`
  - **GPU Version Install** 
    - Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/install/)
    - Create the env based on ```env_phishpedia.yml``` by ```conda env create -f env_phishpedia.yml```
    - ```conda activate env_phishpedia```
    - There are two more env need install ```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```, then ```pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"```
    - Note that, if you use gpu, please comment `cfg.MODEL.DEVICE = 'cpu'` in the `PhishingEval/code/reproduce_phishpedia/train_ob/inference_ob.py`

3. **Prepare Input**. The input should be similar style with `../../data/data_test/data_test.csv`.
4. **Run the Sample**
  - `conda activate env_phishpedia`
  - `python eval_phishpedia.py -siamese_weights=models/bit.pth.tar -targetlist=../../data/targetlist/expand277 -input_csv=../../data/data_test/data_test.csv -input_folder=../../data/data_test`
5. **Run Table 3**. 
  - `conda activate env_phishpedia`
  - `python eval_phishpedia.py -siamese_weights=models/bit.pth.tar -targetlist=../../data/targetlist/expand277 -input_csv=../../data/phishing4190/phishing4190_2.csv -input_folder=../../data/phishing4190`

Note that, I move data_test/ to under data/ path

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
│── AWL/
│── models/
│   ├── demo.pth.tar
│   ├── bit.pth.tar
│   ├── bit_new.pth.tar
│   ├── domain_map.pkl
│   ├── model_final.pth
│   ├── BiT-M-R50x1V2_0.005.pth.tar
│── CRP_Classifier/
│── OCR_Siamese/
│── env_phishintention.yml
│── phishintention_config.py
│── eval_phishintention.py # evaluation file
│── download_model.sh # bash file to download trained models
│── setup_csp.sh # bash file setup the conda env
│── requirement.txt
```
### Preparation
0. **Download Model Weights**
  - `cd PhishingEval/code/reproduce_phishintention`
  - `bash download_model.sh` to download the trained models. The model weights will be saved to `reproduce_phishintention/models`.

1. **Environment** 
  - Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/install/)
  - CPU Env Setup
    - `bash setup_cpu.sh`
    - `conda activate env_phishintention`
  - GPU Env Setup
    - Create the env based on `env_phishintention.yml` by `conda env create -f env_phishintention.yml`
    - `conda activate env_phishintention`
    - There are more env need install `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`, then `pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"`. Then `pip install webdriver-manager==4.0.2`.

2. **Prepare Input**. Input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.
3. **Run the Sample**: 
  - `conda activate env_phishintention`
  - `python eval_phishintention.py -input_csv=../../data/data_test/data_test.csv -input_folder=../../data/data_test -expand=N`
4. **Command to run the phishing4190**: 
  - `conda activate env_phishintention`
  - `python eval_phishintention.py -input_csv=../../data/phishing4190/phishing4190_2.csv -input_folder=../../data/phishing4190 -expand=N`

### Citation
```bibtex
@inproceedings{liu2022inferring,
  title={Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision Based Approach},
  author={Liu, Ruofan and Lin, Yun and Yang, Xianglin and Ng, Siang Hwee and Divakaran, Dinil Mon and Dong, Jin Song},
  booktitle={30th USENIX Security Symposium Security},
  year={2022}
}
```
## run_DynaPhish (I cannot reimplement the installtion. I will try it later)
Original code link [DynaPhish](https://github.com/code-philia/Dynaphish)

**Please focus on other models first, as we are still working on setting up the environment for this one. We have the env for it but can not reinstall again. We'll resolve this issue soon**


- input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.

Since dynaphish is based on PhishIntention, we therefore use our trained phishintention models. Therefore, running this not only use the original one, but also use the `reproduce_phishintention` part.
### Preparation
We based on the original repo to install and replace the phishintention to represuce_phishintention. Will be updated soon due to the environement issue.

1. Google Cloud Part
- Create a google cloud service account, set the billing details
- Enable "Custom Search API", get the API Key and Search Engine ID following this guide.
- Set the [search engine](https://programmablesearchengine.google.com/)
- Create a blank txt file in the directory "knowledge_expansion/api_key.txt", copy and paste your API Key and Search Engine ID into the txt file like the following:
 [YOUR_API_KEY]
 [YOUR_SEARCH_ENGINE_ID]
- For "Cloud Vision API", download the JSON key following this [guide](https://cloud.google.com/sdk/docs/install), save the JSON file under "knowledge_expansion/discoverylabel.json".



<!-- 1. `bash downlaod_model.sh` to download trained models of phishintention
2. ```conda env create -f rundy.yml```
3. ```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```, then ```pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"```. Then ```pip install helium==3.0.9```, ```pip install webdriver-manager==4.0.2```. 
4. ```pip install --no-deps git+https://github.com/lindsey98/PhishIntention.git@development```. 
5. ```git clone https://github.com/lindsey98/MyXdriver_pub.git```. Change the last line of ```setup.sh``` to ```conda run -n "$ENV_NAME" pip install -v .```
6. ```export ENV_NAME="rundy" && bash setup.sh```



- command: ```conda activate rundy``` ->```python -m field_study_logo2brand.dynaphish_main``` -->
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
Original code link [Involution](https://github.com/d-li14/involution)
### CPU Env
0. `cd PhishingEval/code/Involution`
1. `bash setup_cpu.sh`
2. `conda activate env_involution` 
### Run the Sample
1. Crop the logo: 
  - `cd object_detection` 
  - `bash download_model.sh` 
  - `python crop_logo.py -input_folder=../../../data/data_test -input_csv=../../../data/data_test/data_test.csv`
  - Note: please check the screenshot path if appear "NoneType" error
2. Extract the cropped logo info to csv
  - `cd ../involution_paddlepaddle`
  - `python read_crop_logo.py -input_csv=../../../data/data_test/data_test.csv`. The csv file will be saved to `../../../data/data_test/data_test_logo.csv`
3. Eval Sample
  - `under involution_paddlepaddle`
  - `python eval_involution.py -input_csv=../../../data/data_test/data_test_logo.csv -weights=finetune277_models/final.pdparams -targetlist=../../../data/targetlist/expand277 -input_folder=../../../data/data_test` 
### Run Table 3 Results
1. Crop the logo: 
  - `cd object_detection` 
  - `bash download_model.sh` 
  - `python crop_logo.py -input_folder=../../../data/phishing4190 -input_csv=../../../data/phishing4190/phishing4190_2.csv`
  - Note: please check the screenshot path if appear "NoneType" error
2. Extract the cropped logo info to csv
  - `cd ../involution_paddlepaddle`
  - `python read_crop_logo.py -input_csv=../../../data/phishing4190/phishing4190_2.csv`. The csv file will be saved to `../../../data/phishing4190/phishing4190_2_logo.csv`
4. Run Table 3 Results
  - `under involution_paddlepaddle`
  - `python eval_involution.py -input_csv=../../../data/phishing4190/phishing4190_2_logo.csv -weights=finetune277_models/final.pdparams -targetlist=../../../data/targetlist/expand277 -input_folder=../../../data/phishing4190`

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
Original code link [VisualPhishNet](https://github.com/S-Abdelnabi/VisualPhishNet), other reference code link[PhishBaseline](https://github.com/lindsey98/PhishingBaseline)
### Structure
```
VisualPhishnet/
│── env_visualphishnet.yml
│── visualphish_model.py
│── env_visualphishnet.py # evaluation file
│── download_model.sh # bash file to download trained models
```
### Info
- Input: screenshot. Need to be the same style with data_test.csv
### Preparation
1. Make sure you have downloaded the `merge277` or `merge277_new` into the path `PhishingEval/data/targetlist`
2. Download Trained Models:
  - `cd PhishingEval/code/VisualPhishnet`
  - `bash download_model.sh`
3. ENV
  - **CPU ENV**
    - `bash setup_cpu.sh`
    - `conda activate env_visualphishnet`
  - **GPU ENV**
    - `conda crete env -f env_visualphishnet.yaml`
    - `conda activate env_visualphishnet`
    - `pip install scikit-image`
    - `pip install numpy==1.23.5`
4. **Run Sample**: 
  - `conda activate env_visualphishnet`
  - `python eval_visualphishnet.py -targetlist=merge277 -input_folder=../../data/data_test -input_csv=../../data/data_test/data_test.csv`
5. **Running Table3**: 
  - `conda activate env_visualphishnet`
  - `python eval_visualphishnet.py -targetlist=merge277 -input_folder=../../data/phishing4190 -input_csv=../../data/phishing4190/phishing4190_2.csv`

### Citation
```bibtex
@inproceedings{abdelnabi20ccs,
title = {VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity},
author = {Sahar Abdelnabi and Katharina Krombholz and Mario Fritz},
year = {2020},
booktitle = {ACM Conference on Computer and Communications Security (CCS) }
}
```
## PhishZoo
Reference code link [PhishZoo](https://github.com/lindsey98/PhishingBaseline). Input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.
### Preparation
1. `cd PhishingEval/code/PhishZoo`
2. Create env: `bash setup.sh`
3. `conda activate env_phishzoo`
4. **Run Sample**: `python eval_phishzoo.py -targetlist=../../data/targetlist/expand277 -input_csv=../../data/data_test/data_test.csv -input_folder=../../data/data_test`
5. **Run Table3**: `python eval_phishzoo.py -targetlist=../../data/targetlist/expand277 -input_csv=../../data/phishing4190/phishing4190_2.csv -input_folder=../../data/phishing4190`

Note: the dict_construct has a bug that cause the tfidf.csv file always wrong possibility. We will fix it soon. **It will not influence the evaluation for testing sample and Table 3** since the targetlist has already contained the tfidf.csv file. Therefore, we comment the line.
## EMD
Reference code link [EMD](https://github.com/lindsey98/PhishingBaseline). The inputs are screenshots.
### Structure
```
EMD/
│── env_emd.yml
│── eval_emd.py
│── utils.py
```
### Preparation
0. `cd PhishingEval/code/EMD`
1. `bash setup.sh` to create env for EMD. 
2. `conda activate env_emd`
3. **Run Sample**: `python eval_emd.py -targetlist=../../data/targetlist/merge277 -input_csv=../../data/data_test/data_test.csv -input_folder=../../data/data_test`
4. **Run Table 3**: `python eval_emd.py -targetlist=../../data/targetlist/merge277 -input_csv=../../data/phishing4190/phishing4190_2.csv -input_folder=../../data/phishing4190`