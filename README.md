# Code
## reproduce_phishpedia
- Original code link [Phishpedia](https://github.com/lindsey98/Phishpedia)
- input data information: screenshot, url (we use domain in the example code due to the constrain to share urls), and html.
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
## run_DynaPhish
- Original code link [DynaPhish](https://github.com/code-philia/Dynaphish)
## Involution
- Original code link [Involution](https://github.com/d-li14/involution)
## VisualPhishNet
- Original code link [VisualPhishNet](https://github.com/S-Abdelnabi/VisualPhishNet), other reference code link[PhishBaseline](https://github.com/lindsey98/PhishingBaseline)
## PhishZoo
- Reference code link [PhishZoo](https://github.com/lindsey98/PhishingBaseline)
## EMD
- Reference code link [EMD](https://github.com/lindsey98/PhishingBaseline)

# dataset
- apwg451514: contains html, screenshots from apwg. Will be shared on the otehr website due to its large size
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

