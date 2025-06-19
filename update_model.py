# update_model.py

import pandas as pd
import os
import joblib
import logging
import urllib.parse
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from utils import download_file_from_gdrive

# --- লগিং কনফিগারেশন ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers or len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.FileHandler):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- ডেটাবেস এবং গুগল ড্রাইভ কানেকশন ---
MONGO_USERNAME = os.environ.get("MONGO_USERNAME", "mdhasannirob271")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "Odjlfg1XtZqcjx32")
MONGO_CLUSTER_URL = os.environ.get("MONGO_CLUSTER_URL", "hatespeech.u6hagva.mongodb.net")
DB_NAME = "hatespeech_db"
COLLECTION_NAME = "wrong_predictions"

BENGALI_DATA_URL = "https://drive.google.com/uc?export=download&id=1GorLcJrmwo-RQmpBaaugdjU1nZayOKBZ"
ENGLISH_DATA_URL = "https://drive.google.com/uc?export=download&id=1UdU6HqyL57ZK9-uU26gkf92zl7a4gL7T"
MIN_REPORTS_TO_RETRAIN = 10

# --- ডেটাবেস ক্লায়েন্ট ---
try:
    encoded_password = urllib.parse.quote_plus(MONGO_PASSWORD)
    CONNECTION_STRING = f"mongodb+srv://{MONGO_USERNAME}:{encoded_password}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&appName=hatespeech"
    client = MongoClient(CONNECTION_STRING)
    client.admin.command('ping')
    db = client[DB_NAME]
    reports_collection = db[COLLECTION_NAME]
    logger.info("✅ (Update Script) MongoDB-এর সাথে সংযুক্ত।")
except Exception as e:
    logger.error(f"❌ (Update Script) MongoDB সংযোগ ব্যর্থ: {e}")
    client = None

# --- Helper Functions ---
def get_stop_words(lang='bn'):
    # ... (এই ফাংশনটি আগের মতোই থাকবে) ...

def train_single_model(texts, labels, lang_code):
    # ... (এই ফাংশনটি আগের মতোই থাকবে) ...

# --- প্রধান আপডেট ফাংশন ---
def update_models_from_feedback():
    if client is None:
        logger.error("ডেটাবেস সংযোগ নেই। আপডেট প্রক্রিয়া চালানো সম্ভব নয়।")
        return

    logger.info("===== মডেল আপডেট প্রক্রিয়া শুরু হচ্ছে =====")
    
    # ডেটাবেস থেকে সব রিপোর্ট পড়া
    new_reports = list(reports_collection.find({}))

    if not new_reports or len(new_reports) < MIN_REPORTS_TO_RETRAIN:
        logger.info(f"ডেটাবেসে পর্যাপ্ত নতুন রিপোর্ট নেই (প্রয়োজনীয়: {MIN_REPORTS_TO_RETRAIN})।")
        return

    reports_df = pd.DataFrame(new_reports)
    logger.info(f"{len(reports_df)} টি নতুন রিপোর্ট ডেটাবেস থেকে পাওয়া গেছে।")

    # ভাষা অনুযায়ী রিপোর্ট ভাগ করা
    reports_by_lang = {'bn': [], 'en': []}
    for _, row in reports_df.iterrows():
        try:
            lang = detect(row['text'])
            if lang in reports_by_lang:
                reports_by_lang[lang].append(row)
        except LangDetectException:
            logger.warning(f"'{row['text'][:20]}...' এর ভাষা শনাক্ত করা যায়নি।")
    
    # প্রতিটি ভাষার জন্য মডেল আপডেট করা
    updated_something = False
    for lang_code, reports in reports_by_lang.items():
        if not reports: continue

        logger.info(f"'{lang_code}' ভাষার জন্য {len(reports)} টি রিপোর্ট পাওয়া গেছে। মডেল আপডেট করা হবে।")
        updated_something = True
        
        dataset_url = BENGALI_DATA_URL if lang_code == 'bn' else ENGLISH_DATA_URL
        dataset_path = os.path.join('data', f'Cleaned_{"Bangla" if lang_code == "bn" else "english"}_hatespeech.csv')

        if not os.path.exists(dataset_path):
            if not download_file_from_gdrive(dataset_url, dataset_path):
                logger.error(f"'{dataset_path}' ডাউনলোড ব্যর্থ। '{lang_code}' মডেল আপডেট করা সম্ভব নয়।")
                continue

        main_df = pd.read_csv(dataset_path)
        new_data_df = pd.DataFrame(reports).rename(columns={'text': 'texts', 'correct_label': 'labels'})

        combined_df = pd.concat([main_df, new_data_df[['texts', 'labels']]], ignore_index=True)
        combined_df.drop_duplicates(subset=['texts'], keep='last', inplace=True)
        combined_df.to_csv(dataset_path, index=False, encoding='utf-8')
        logger.info(f"'{dataset_path}' আপডেট করা হয়েছে।")

        train_single_model(combined_df['texts'].astype(str), combined_df['labels'].astype(int), lang_code)

    # যদি কোনো মডেল আপডেট হয়ে থাকে, তবেই ডেটাবেস থেকে রিপোর্ট ডিলিট করা হবে
    if updated_something:
        report_ids_to_delete = [report['_id'] for report in new_reports]
        if report_ids_to_delete:
            reports_collection.delete_many({'_id': {'$in': report_ids_to_delete}})
            logger.info("ডেটাবেস থেকে প্রসেস করা রিপোর্টগুলো ডিলিট করা হয়েছে।")
    
    logger.info("===== মডেল আপডেট প্রক্রিয়া সম্পন্ন! =====")

if __name__ == '__main__':
    # ফোল্ডার তৈরি করা (যদি না থাকে)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if client:
        update_models_from_feedback()
