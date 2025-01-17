# used_models
- Due to its large size, we share the trained models on [this link](https://liveutk-my.sharepoint.com/:f:/g/personal/fji1_vols_utk_edu/EiDwgElIisBAjA7H5LUnAL0BZQFdvtbTjXR_c03MWsKkgw?e=6CC3dm)
# Code
## reproduce_phishpedia
- Original code link [Phishpedia](https://github.com/lindsey98/Phishpedia)
- input data information: screenshot and url (we use domain in the example code due to the constrain to share urls).
- conda env: please install conda env based on original github env, and then follow the version of env_phishpedia.yml
- download the trained_models folder and put it under repreduce_phishpedia. 
- paths: model path, reference list, etc. is in "config.yaml"
- command to run the code: ```conda activate env_phishpedia``` -> ```python eval_phishpedia.py```
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
# dataset
- Note: Due to its large size, we put the subset of sampled phishing 4190 dataset corresponding to Table 3 in the paper. The apwg451514 is shared through the other link.
- apwg451514: contains html, screenshots from apwg. Due to its large size, we share it at [this link]()
- archive100: 100 domain (Tranco1000) with html and screenshots (archive.org)
- crawl_benign: benign 110 brands' data, including:
    - login.png: screenshot
    - login.txt: url
    - login.html: html
    - classes.txt: class for label
    - XXX-login.txt: logo region
- failed_example_csv: the csv contains the html and screenshot paths that let models fail. Extract screenshot and html from apwg451514
- perturbated_dataset: black box attack and white box attach
- targetlist: reference lists
    - expand277: PhishIntention-based logo reference list
    - expand277_new: expanded logo-based logo reference list
    - merge277: screenshot-based logo reference list
    - merge277_new: expanded screenshot-based logo reference list
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

