import os, sys
import time
import json
from datetime import datetime

from multiprocessing.pool import ThreadPool, Pool
import threading

import requests
import urllib3
from urllib.parse import urljoin
from urllib.parse import urlparse
import tldextract
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


import pandas as pd
import random

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def save_page(page_source, current_url, session, page_filename='index'):
    try:
        save_file_name = "index" + ".html"
        with open(os.path.join(root_path, save_file_name), 'wb') as file:
            file.write(page_source.encode('utf-8')) 
    except Exception as e:
        print("error-save-page", e)

def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument("--disable-client-side-phishing-detection")
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1280, 960)
    driver.set_page_load_timeout(100)
    return driver


def take_screenshot(target_url, mode, retries=2):
    """
    if None is returned, accessing the webapge is not successful
    Otherwise, page_source is returned
    """
    time.sleep(random.uniform(10, 30))
    for attempt in range(retries):
        driver = None
        time.sleep(3)
        try:
            driver = create_webdriver()
            driver.get(target_url)
            current_url = driver.current_url
            page_source = driver.page_source
            
            driver.save_screenshot(os.path.join(root_path, "index-screenshot"+  ".png"))

            return [page_source, current_url]
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                return [None, None]
        finally:
            if driver:
                driver.quit()
    
def apwg_scrapping(target_url):
    time.sleep(random.uniform(10, 30))
    
    
    os.makedirs(root_path, exist_ok=True)
    files = os.listdir(root_path)
            
    # Check for files containing 'screenshot' in their names
    screenshot_files = [file for file in files if 'screenshot-desktop' in file.lower()]
    html_files = [file for file in files if 'index.html' in file.lower()]
    
    if (len(screenshot_files) > 0) and (len(html_files) > 0):
        return
    with open(os.path.join(root_path, "info.json"), "w") as f:
        json.dump(target_url, f)

    try:
        r = requests.get(target_url, timeout=15, headers=requests_headers["desktop"], verify=False)
    except Exception as e:
        print("Error-apwg_scrapping", e)
        return

    desktop_page_source = ""
    for mode in ["desktop"]:
        [page_source, current_url] = take_screenshot(target_url, mode)
        if (mode == "desktop") and (page_source is None):
            break
        
        if page_source != None: # if success
            # print(id, "current_url:", current_url)
            session = requests.Session()
            try:
                
                save_page(page_source, current_url, session)
            
            except Exception as e:
                print("error", id, e, mode)


if __name__ == "__main__":
    
    requests_headers = {
        "desktop": {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        "mobile": {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'}
    }

    root_path = "crawl" # the folder path that save the crawled info

    url = "https://www.google.com"

    apwg_scrapping(url)
            
        
