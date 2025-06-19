# update_model.py

import pandas as pd
import os
import joblib
import logging
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- লগিং এবং ধ্রুবক ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WRONG_PREDICTIONS_FILE = os.path.join('data', 'wrong_predictions.csv')
MIN_REPORTS_TO_RETRAIN = 10 # কমপক্ষে ১০টি রিপোর্ট জমা হলে আপডেট হবে

# --- Helper Functions (train_model.py থেকে আনা) ---
def get_stop_words(lang='bn'):
    if lang == 'en':
        return ["a", "an", "the", "in", "on", "is", "are", "i", "you", "he", "she", "it"] # উদাহরণ
    # ডিফল্ট বাংলা
    return ["এবং", "অথবা", "কিন্তু", "যদি", "তবে", "আমি", "তুমি", "সে", "আমরা", "তোমরা"] # উদাহরণ

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

    if not os.path.exists(WRONG_PREDICTIONS_FILE):
        logger.info("কোনো নতুন রিপোর্ট ফাইল নেই।")
        return

    try:
        reports_df = pd.read_csv(WRONG_PREDICTIONS_FILE)
    except pd.errors.EmptyDataError:
        logger.info("রিপোর্ট ফাইলটি খালি।")
        return

    if len(reports_df) < MIN_REPORTS_TO_RETRAIN:
        logger.info(f"পর্যাপ্ত নতুন রিপোর্ট জমা হয়নি (প্রয়োজনীয়: {MIN_REPORTS_TO_RETRAIN}, পাওয়া গেছে: {len(reports_df)})।")
        return

    logger.info(f"{len(reports_df)} টি নতুন রিপোর্ট পাওয়া গেছে। ভাষা অনুযায়ী ভাগ করা হচ্ছে...")

    # ভাষা অনুযায়ী রিপোর্টগুলো আলাদা করা
    reports_by_lang = {'bn': [], 'en': []}
    for _, row in reports_df.iterrows():
        try:
            lang = detect(row['text'])
            if lang in reports_by_lang:
                reports_by_lang[lang].append(row)
        except LangDetectException:
            logger.warning(f"'{row['text'][:20]}...' এর ভাষা শনাক্ত করা যায়নি। উপেক্ষা করা হচ্ছে।")
    
    # প্রতিটি ভাষার জন্য মডেল আপডেট করা
    for lang_code, reports in reports_by_lang.items():
        if not reports:
            logger.info(f"'{lang_code}' ভাষার জন্য কোনো নতুন রিপোর্ট নেই।")
            continue

        report_count = len(reports)
        logger.info(f"'{lang_code}' ভাষার জন্য {report_count} টি রিপোর্ট পাওয়া গেছে। এই ভাষার মডেল আপডেট করা হবে।")

        # নির্দিষ্ট ভাষার মূল ডেটাসেট লোড করা
        lang_dataset_file = os.path.join('data', f'Cleaned_Bangla_hatespeech.csv' if lang_code == 'bn' else 'english_hatespeech_cleaned.csv')
        if not os.path.exists(lang_dataset_file):
            logger.error(f"'{lang_dataset_file}' খুঁজে পাওয়া যায়নি। '{lang_code}' মডেল আপডেট করা সম্ভব নয়।")
            continue

        main_df = pd.read_csv(lang_dataset_file)
        new_data_df = pd.DataFrame(reports).rename(columns={'text': 'texts', 'correct_label': 'labels'})

        # ডেটা একত্রিত এবং ডুপ্লিকেট সরানো
        combined_df = pd.concat([main_df, new_data_df[['texts', 'labels']]], ignore_index=True)
        combined_df.drop_duplicates(subset=['texts'], keep='last', inplace=True)
        
        # নতুন ডেটা সহ মূল ডেটাসেট ফাইল আপডেট করা
        combined_df.to_csv(lang_dataset_file, index=False, encoding='utf-8')
        logger.info(f"'{lang_dataset_file}' আপডেট করা হয়েছে।")

        # মডেল রি-ট্রেইন করা
        train_single_model(combined_df['texts'].astype(str), combined_df['labels'].astype(int), lang_code)

    # সবশেষে, রিপোর্ট ফাইলটি খালি করে দেওয়া
    try:
        os.remove(WRONG_PREDICTIONS_FILE)
        logger.info(f"রিপোর্ট ফাইল '{WRONG_PREDICTIONS_FILE}' সফলভাবে খালি করা হয়েছে।")
    except OSError as e:
        logger.error(f"রিপোর্ট ফাইল ডিলিট করতে সমস্যা: {e}")

    logger.info("===== মডেল আপডেট প্রক্রিয়া সম্পন্ন! =====")

if __name__ == '__main__':
    update_models_from_feedback()