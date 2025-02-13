import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
import fastdup
import csv
from pathlib import Path

if __name__ == "__main__":
    
    # 
    fd = fastdup.create(input_dir="XXXX", work_dir="XXXX")
    fd.run(ccthreshold=0.96, license="XXXXXX")# overwrite=True, 
    fd.summary()
    fd.vis.duplicates_gallery(num_images=5)
    fd.vis.component_gallery(num_images=5)
