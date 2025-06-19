# update_model.py

import pandas as pd
import os
import joblib
import logging
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import download_file_from_gdrive # <-- নতুন ইম্পোর্ট

# --- লগিং এবং ধ্রুবক ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers or len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.FileHandler):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

BENGALI_DATA_URL = "https://drive.google.com/uc?export=download&id=1GorLcJrmwo-RQmpBaaugdjU1nZayOKBZ"
ENGLISH_DATA_URL = "https://drive.google.com/uc?export=download&id=1UdU6HqyL57ZK9-uU26gkf92zl7a4gL7T"
WRONG_PREDICTIONS_FILE = os.path.join('data', 'wrong_predictions.csv')
MIN_REPORTS_TO_RETRAIN = 10

# --- Helper Functions ---
def get_stop_words(lang='bn'):
    if lang == 'en':
        return ["a", "an", "the", "in", "on", "is", "are", "i", "you", "he", "she", "it"]
    return ["এবং", "অথবা", "কিন্তু", "যদি", "তবে", "আমি", "তুমি", "সে", "আমরা", "তোমরা"]

def train_single_model(texts, labels, lang_code):
    """ একটি নির্দিষ্ট ভাষার জন্য মডেল ট্রেইন করে এবং সেভ করে। """
    logger.info(f"'{lang_code}' ভাষার জন্য মডেল নতুন করে ট্রেইন করা হচ্ছে...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words=get_stop_words(lang_code), min_df=3, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', C=10.0, solver='liblinear')
    model.fit(X, labels)
    model_path = os.path.join('models', f'model_{lang_code}.joblib')
    vectorizer_path = os.path.join('models', f'vectorizer_{lang_code}.joblib')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"✅ '{lang_code}' ভাষার মডেল সফলভাবে আপডেট এবং সেভ করা হয়েছে।")

# --- প্রধান আপডেট ফাংশন ---
def update_models_from_feedback():
    logger.info("===== মডেল আপডেট প্রক্রিয়া শুরু হচ্ছে =====")

    if not os.path.exists(WRONG_PREDICTIONS_FILE) or os.path.getsize(WRONG_PREDICTIONS_FILE) == 0:
        logger.info("কোনো নতুন রিপোর্ট ফাইল নেই। আপডেট করার প্রয়োজন নেই।")
        return

    try:
        reports_df = pd.read_csv(WRONG_PREDICTIONS_FILE)
    except pd.errors.EmptyDataError:
        logger.info("রিপোর্ট ফাইলটি খালি। আপডেট করার প্রয়োজন নেই।")
        return

    if len(reports_df) < MIN_REPORTS_TO_RETRAIN:
        logger.info(f"পর্যাপ্ত নতুন রিপোর্ট জমা হয়নি (প্রয়োজনীয়: {MIN_REPORTS_TO_RETRAIN}, পাওয়া গেছে: {len(reports_df)})।")
        return

    logger.info(f"{len(reports_df)} টি নতুন রিপোর্ট পাওয়া গেছে। ভাষা অনুযায়ী ভাগ করা হচ্ছে...")

    reports_by_lang = {'bn': [], 'en': []}
    for _, row in reports_df.iterrows():
        try:
            lang = detect(row['text'])
            if lang in reports_by_lang:
                reports_by_lang[lang].append(row)
        except LangDetectException:
            logger.warning(f"'{row['text'][:20]}...' এর ভাষা শনাক্ত করা যায়নি। উপেক্ষা করা হচ্ছে।")
    
    for lang_code, reports in reports_by_lang.items():
        if not reports:
            logger.info(f"'{lang_code}' ভাষার জন্য কোনো নতুন রিপোর্ট নেই।")
            continue

        logger.info(f"'{lang_code}' ভাষার জন্য {len(reports)} টি রিপোর্ট পাওয়া গেছে। এই ভাষার মডেল আপডেট করা হবে।")

        # --- === পরিবর্তন এখানে === ---
        # নির্দিষ্ট ভাষার মূল ডেটাসেট লোড করা
        dataset_url = BENGALI_DATA_URL if lang_code == 'bn' else ENGLISH_DATA_URL
        dataset_path = os.path.join('data', f'Cleaned_{"Bangla" if lang_code == "bn" else "english"}_hatespeech.csv')

        # যদি মূল ডেটাসেট না থাকে, তবে গুগল ড্রাইভ থেকে ডাউনলোড করা
        if not os.path.exists(dataset_path):
            logger.info(f"'{dataset_path}' খুঁজে পাওয়া যায়নি। গুগল ড্রাইভ থেকে ডাউনলোড করা হচ্ছে...")
            if not download_file_from_gdrive(dataset_url, dataset_path):
                logger.error(f"'{dataset_path}' ডাউনলোড ব্যর্থ। '{lang_code}' মডেল আপডেট করা সম্ভব নয়।")
                continue # পরের ভাষার জন্য চেষ্টা করা
        # --- === পরিবর্তন শেষ === ---

        main_df = pd.read_csv(dataset_path)
        new_data_df = pd.DataFrame(reports).rename(columns={'text': 'texts', 'correct_label': 'labels'})

        combined_df = pd.concat([main_df, new_data_df[['texts', 'labels']]], ignore_index=True)
        combined_df.drop_duplicates(subset=['texts'], keep='last', inplace=True)
        combined_df.to_csv(dataset_path, index=False, encoding='utf-8')
        logger.info(f"'{dataset_path}' আপডেট করা হয়েছে।")

        train_single_model(combined_df['texts'].astype(str), combined_df['labels'].astype(int), lang_code)

    try:
        os.remove(WRONG_PREDICTIONS_FILE)
        logger.info(f"রিপোর্ট ফাইল '{WRONG_PREDICTIONS_FILE}' সফলভাবে খালি করা হয়েছে।")
    except OSError as e:
        logger.error(f"রিপোর্ট ফাইল ডিলিট করতে সমস্যা: {e}")

    logger.info("===== মডেল আপডেট প্রক্রিয়া সম্পন্ন! =====")

if __name__ == '__main__':
    # ফোল্ডার তৈরি করা (যদি না থাকে)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    update_models_from_feedback()
