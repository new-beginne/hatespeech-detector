# train_model_en.py

import os
import csv
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import download_file_from_gdrive # <-- নতুন ইম্পোর্ট

# --- লগিং কনফিগারেশন ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers or len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.FileHandler):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def prepare_dataset(csv_file):
    """CSV ফাইল থেকে ইংরেজি ডেটাসেট লোড করে।"""
    texts, labels = [], []
    logger.info(f"'{csv_file}' থেকে ইংরেজি ডেটাসেট লোড করা হচ্ছে...")
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'texts' in row and 'labels' in row:
                    texts.append(row['texts'])
                    labels.append(int(row['labels']))
        
        logger.info(f"ইংরেজি ডেটাসেট লোড সম্পন্ন। মোট নমুনা: {len(texts)}")
        return texts, labels
    except FileNotFoundError:
        logger.error(f"ত্রুটি: '{csv_file}' ফাইলটি খুঁজে পাওয়া যায়নি।")
        return [], []

def get_english_stop_words():
    """ইংরেজি স্টপ ওয়ার্ডের তালিকা।"""
    return [
        "a", "an", "the", "in", "on", "is", "are", "i", "me", "my", "myself", 
        "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", 
        "we", "us", "our", "ours", "they", "them", "their", "theirs", "to", "of", 
        "for", "and", "or", "but", "if", "because", "as", "until", "while", 
        "at", "by", "with", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "from", "up", "down", 
        "out", "over", "under", "again", "further", "then", "once", "here", 
        "there", "when", "where", "why", "how", "all", "any", "both", "each", 
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
        "will", "just", "don", "should", "now"
    ]

def train_english_model(csv_file):
    """ইংরেজি ডেটাসেটের উপর মডেল ট্রেইন করে এবং _en.joblib হিসেবে সেভ করে।"""
    try:
        logger.info("===== ইংরেজি মডেল ট্রেইনিং প্রক্রিয়া শুরু হচ্ছে =====")
        texts, labels = prepare_dataset(csv_file)
        
        if not texts or len(texts) < 100:
            logger.error("ট্রেইনিং বন্ধ করা হচ্ছে। পর্যাপ্ত ডেটা নেই।")
            return

        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            stop_words=get_english_stop_words(),
            min_df=5,
            max_df=0.7
        )
        X = vectorizer.fit_transform(texts)
        logger.info(f"ইংরেজি ডেটার ভেক্টর আকার: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=5.0,
            solver='liblinear'
        )
        model.fit(X_train, y_train)
        
        logger.info("===== ইংরেজি মডেল ইভ্যালুয়েশন =====")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Hate/Offensive'], zero_division=0)
        
        logger.info(f"ইংরেজি মডেলের অ্যাকুরেসি: {accuracy:.2%}")
        logger.info("ক্লাসিফিকেশন রিপোর্ট:\n" + report)
        
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'model_en.joblib')
        vectorizer_path = os.path.join('models', 'vectorizer_en.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"ইংরেজি মডেল সফলভাবে '{model_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info(f"ইংরেজি ভেক্টরাইজার সফলভাবে '{vectorizer_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info("===== ইংরেজি মডেল ট্রেইনিং প্রক্রিয়া সম্পন্ন =====")

    except Exception as e:
        logger.error(f"ইংরেজি মডেল ট্রেইনিং এর সময় একটি ত্রুটি ঘটেছে: {e}", exc_info=True)

# --- === প্রধান অংশ (পরিবর্তন আনা হয়েছে) === ---
if __name__ == '__main__':
    # ফোল্ডার তৈরি করা
    os.makedirs('data', exist_ok=True)

    # গুগল ড্রাইভ থেকে ইংরেজি ডেটাসেট ডাউনলোড করার লিংক
    ENGLISH_DATASET_URL = "https://drive.google.com/uc?export=download&id=1UdU6HqyL57ZK9-uU26gkf92zl7a4gL7T"
    ENGLISH_DATA_PATH = os.path.join('data', 'english_hatespeech_cleaned.csv')

    # ধাপ ১: ডেটাসেট ডাউনলোড করা
    if not os.path.exists(ENGLISH_DATA_PATH):
        logger.info(f"লোকাল মেশিনে '{ENGLISH_DATA_PATH}' পাওয়া যায়নি। গুগল ড্রাইভ থেকে ডাউনলোড করা হচ্ছে...")
        if not download_file_from_gdrive(ENGLISH_DATASET_URL, ENGLISH_DATA_PATH):
            logger.error("ডেটাসেট ডাউনলোড ব্যর্থ। ট্রেইনিং বন্ধ করা হচ্ছে।")
            exit()
    else:
        logger.info(f"'{ENGLISH_DATA_PATH}' ফাইলটি লোকাল মেশিনে পাওয়া গেছে। ডাউনলোড করার প্রয়োজন নেই।")
        
    # ধাপ ২: মডেল ট্রেইন করা
    train_english_model(ENGLISH_DATA_PATH)
