# utils.py

import requests
import os
import logging

def download_file_from_gdrive(url, output_path):
    """গুগল ড্রাইভ থেকে ফাইল ডাউনলোড করে নির্দিষ্ট পাথে সেভ করে।"""
    try:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        logging.info(f"'{output_path}' ফাইলে ডেটাসেট ডাউনলোড করা হচ্ছে...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"ডাউনলোড সম্পন্ন: {output_path}")
        return True
    except Exception as e:
        logging.error(f"ডেটাসেট ডাউনলোড করতে ত্রুটি: {e}")
        return False