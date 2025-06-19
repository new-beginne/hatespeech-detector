# update_data.py

import pandas as pd
import os
import logging
from langdetect import detect, LangDetectException
from pymongo import MongoClient
import urllib.parse
from utils import read_sheet_as_df, overwrite_sheet_with_df # নতুন ইম্পোর্ট

# --- কনফিগারেশন ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MONGO_USERNAME = os.environ.get("MONGO_USERNAME", "mdhasannirob271")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "Odjlfg1XtZqcjx32")
MONGO_CLUSTER_URL = os.environ.get("MONGO_CLUSTER_URL", "hatespeech.u6hagva.mongodb.net")
DB_NAME = "hatespeech_db"
COLLECTION_NAME = "wrong_predictions"
MIN_REPORTS_TO_RETRAIN = 10 

# ডেটাবেস কানেকশন
encoded_password = urllib.parse.quote_plus(MONGO_PASSWORD)
CONNECTION_STRING = f"mongodb+srv://{MONGO_USERNAME}:{encoded_password}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&appName=hatespeech"
client = MongoClient(CONNECTION_STRING)
db = client[DB_NAME]
reports_collection = db[COLLECTION_NAME]

def update_datasets_with_feedback():
    logger.info("===== ডেটাসেট আপডেট প্রক্রিয়া শুরু হচ্ছে =====")
    
    new_reports = list(reports_collection.find({}))
    if not new_reports or len(new_reports) < MIN_REPORTS_TO_RETRAIN:
        logger.info("ডেটাবেসে পর্যাপ্ত নতুন রিপোর্ট নেই।")
        return

    reports_df = pd.DataFrame(new_reports)
    
    reports_by_lang = {'bn': [], 'en': []}
    for _, row in reports_df.iterrows():
        try:
            lang = detect(row['text'])
            if lang in reports_by_lang: reports_by_lang[lang].append(row)
        except LangDetectException: pass

    updated_something = False
    for lang_code, reports in reports_by_lang.items():
        if not reports: continue

        sheet_name = "Cleaned_Bangla_hatespeech" if lang_code == 'bn' else "english_hatespeech_cleaned"
        main_df = read_sheet_as_df(sheet_name)
        if main_df is None:
            logger.error(f"'{sheet_name}' পড়া সম্ভব হয়নি। এই ভাষার জন্য আপডেট বন্ধ।")
            continue

        new_data_df = pd.DataFrame(reports).rename(columns={'text': 'texts', 'correct_label': 'labels'})
        
        combined_df = pd.concat([main_df, new_data_df[['texts', 'labels']]], ignore_index=True)
        combined_df.drop_duplicates(subset=['texts'], keep='last', inplace=True)
        
        if overwrite_sheet_with_df(combined_df, sheet_name):
            updated_something = True

    if updated_something:
        report_ids_to_delete = [report['_id'] for report in new_reports]
        if report_ids_to_delete:
            reports_collection.delete_many({'_id': {'$in': report_ids_to_delete}})
            logger.info("ডেটাবেস থেকে প্রসেস করা রিপোর্টগুলো ডিলিট করা হয়েছে।")
            
    logger.info("===== ডেটাসেট আপডেট প্রক্রিয়া সম্পন্ন =====")

if __name__ == '__main__':
    update_datasets_with_feedback()
