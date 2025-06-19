# train_model_en.py

import os
import csv
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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
            C=5.0, # ইংরেজি ভাষার জন্য C-এর মান কিছুটা ভিন্ন হতে পারে
            solver='liblinear'
        )
        model.fit(X_train, y_train)
        
        logger.info("===== ইংরেজি মডেল ইভ্যালুয়েশন =====")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Hate/Offensive'])
        
        logger.info(f"ইংরেজি মডেলের অ্যাকুরেসি: {accuracy:.2%}")
        logger.info("ক্লাসিফিকেশন রিপোর্ট:\n" + report)
        
        # মডেল এবং ভেক্টরাইজার _en.joblib হিসেবে সেভ করা
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'model_en.joblib')
        vectorizer_path = os.path.join('models', 'vectorizer_en.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"ইংরেজি মডেল সফলভাবে '{model_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info(f"ইংরেজি ভেক্টরাইজার সফলভাবে '{vectorizer_path}' ফাইলে সেভ করা হয়েছে।")

    except Exception as e:
        logger.error(f"ইংরেজি মডেল ট্রেইনিং এর সময় একটি ত্রুটি ঘটেছে: {e}", exc_info=True)

if __name__ == '__main__':
    # প্রস্তুত করা ইংরেজি ডেটাসেট ফাইলের পাথ
    english_csv_path = os.path.join('data', 'english_hatespeech_cleaned.csv')
    
    # ইংরেজি মডেল ট্রেইনিং ফাংশন কল করা
    train_english_model(english_csv_path)