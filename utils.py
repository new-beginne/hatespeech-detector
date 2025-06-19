# utils.py

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
import os
import json

# গুগল ক্রেডেনশিয়াল লোড করা
def get_gspread_client():
    """
    Render-এর এনভায়রনমেন্ট ভ্যারিয়েবল অথবা লোকাল credentials.json ফাইল থেকে
    গুগল ক্রেডেনশিয়াল লোড করে এবং gspread ক্লায়েন্ট রিটার্ন করে।
    """
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
    
    # Render এনভায়রনমেন্ট থেকে পড়ার চেষ্টা
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS")
    if creds_json_str:
        logging.info("Render এনভায়রনমেন্ট ভ্যারিয়েবল থেকে গুগল ক্রেডেনশিয়াল লোড করা হচ্ছে...")
        creds_dict = json.loads(creds_json_str)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    # লোকাল ফাইল থেকে পড়ার চেষ্টা
    elif os.path.exists('credentials.json'):
        logging.info("লোকাল 'credentials.json' ফাইল থেকে গুগল ক্রেডেনশিয়াল লোড করা হচ্ছে...")
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    else:
        logging.error("গুগল ক্রেডেনশিয়াল খুঁজে পাওয়া যায়নি!")
        return None
        
    return gspread.authorize(creds)

def read_sheet_as_df(sheet_name):
    """ একটি গুগল শিটকে নাম দিয়ে খুলে pandas DataFrame হিসেবে রিটার্ন করে। """
    try:
        gc = get_gspread_client()
        if not gc: return None
        
        worksheet = gc.open(sheet_name).sheet1
        logging.info(f"'{sheet_name}' গুগল শিট সফলভাবে পড়া হয়েছে।")
        return pd.DataFrame(worksheet.get_all_records())
    except Exception as e:
        logging.error(f"'{sheet_name}' গুগল শিট পড়তে সমস্যা: {e}")
        return None

def overwrite_sheet_with_df(df, sheet_name):
    """ একটি pandas DataFrame দিয়ে সম্পূর্ণ গুগল শিটকে ওভাররাইট করে। """
    try:
        gc = get_gspread_client()
        if not gc: return False

        worksheet = gc.open(sheet_name).sheet1
        worksheet.clear()
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        logging.info(f"'{sheet_name}' গুগল শিট সফলভাবে আপডেট করা হয়েছে।")
        return True
    except Exception as e:
        logging.error(f"'{sheet_name}' গুগল শিটে লিখতে সমস্যা: {e}")
        return False
