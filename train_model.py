import os
import re
import csv
import joblib
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import download_file_from_gdrive # <-- নতুন ইম্পোর্ট

# --- লগিং কনফিগারেশন ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='training.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def prepare_dataset(csv_file):
    """CSV ফাইল থেকে ডেটাসেট লোড এবং ভ্যালিডেট করে।"""
    texts, labels = [], []
    logger.info(f"'{csv_file}' থেকে ডেটাসেট লোড করা হচ্ছে...")
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text_col = 'texts' if 'texts' in row else 'text'
                label_col = 'labels' if 'labels' in row else 'label'

                if text_col in row and label_col in row:
                    text = row[text_col].strip()
                    label = row[label_col].strip()
                    if text and label in ['0', '1']:
                        texts.append(text)
                        labels.append(int(label))
        
        if not texts:
            logger.error("ডেটাসেটে কোনো বৈধ ডেটা পাওয়া যায়নি।")
            return [], []

        total, hate_count = len(texts), sum(labels)
        normal_count = total - hate_count
        logger.info(f"ডেটাসেট লোড সম্পন্ন। পরিসংখ্যান: মোট={total}, হ্যাট={hate_count} ({hate_count/total:.1%}), নরমাল={normal_count} ({normal_count/total:.1%})")
        return texts, labels
    except FileNotFoundError:
        logger.error(f"ত্রুটি: '{csv_file}' ফাইলটি খুঁজে পাওয়া যায়নি।")
        return [], []
    except Exception as e:
        logger.error(f"ডেটাসেট লোড করতে একটি অপ্রত্যাশিত ত্রুটি ঘটেছে: {e}")
        return [], []

def get_stop_words():
    """শুধুমাত্র সাধারণ বাংলা স্টপ ওয়ার্ড রিটার্ন করে।"""
    stop_words = {
        "এবং", "অথবা", "কিন্তু", "যদি", "তবে", "আমি", "তুমি", "সে", "আমরা", "তোমরা", 
        "তারা", "এই", "যে", "কি", "না", "হ্যাঁ", "তো", "কে", "আর", "ও", "নেই", "হবে", 
        "হয়", "করতে", "করবে", "যাও", "গেলে", "থেকে", "পর্যন্ত", "চাই", "করা", "তাহলে",
        "করে", "হল", "হলে", "যেমন", "তেমন", "সব", "এক", "দুই", "করেছে", "নয়", "হত",
        "তা", "তাই", "ওই", "এটা", "সেটা", "এখন", "আগে", "পরে", "মধ্যে"
    }
    return list(stop_words)

def train_bengali_model(csv_file): # <-- ফাংশনের নাম পরিবর্তন করা হয়েছে
    """বাংলা মডেল ট্রেইন, ইভ্যালুয়েট এবং সেভ করার মূল ফাংশন।"""
    try:
        logger.info("===== বাংলা মডেল ট্রেইনিং প্রক্রিয়া শুরু হচ্ছে =====")
        texts, labels = prepare_dataset(csv_file)
        
        if not texts or len(texts) < 100:
            logger.error("ট্রেইনিং বন্ধ করা হচ্ছে। ডেটাসেটে পর্যাপ্ত ডেটা নেই (< 100)।")
            return

        logger.info("ফিচার ভেক্টরাইজেশন শুরু হচ্ছে...")
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words=get_stop_words(),
            min_df=3,
            max_df=0.9,
            token_pattern=r'(?u)\b\w+\b'
        )
        X = vectorizer.fit_transform(texts)
        logger.info(f"ভেক্টরাইজড ডেটা আকার: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        logger.info("মডেল ট্রেইনিং শুরু হচ্ছে...")
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=10.0,
            solver='liblinear',
            penalty='l2',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        logger.info("===== মডেল ইভ্যালুয়েশন =====")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Hate'], zero_division=0)
        
        logger.info(f"মডেলের অ্যাকুরেসি: {accuracy:.2%}")
        logger.info("ক্লাসিফিকেশন রিপোর্ট:\n" + report)
        
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'model_bn.joblib') # <-- ফাইলের নাম পরিবর্তন
        vectorizer_path = os.path.join('models', 'vectorizer_bn.joblib') # <-- ফাইলের নাম পরিবর্তন
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"মডেল সফলভাবে '{model_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info(f"ভেক্টরাইজার সফলভাবে '{vectorizer_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info("===== বাংলা মডেল ট্রেইনিং প্রক্রিয়া সম্পন্ন =====")
        
    except Exception as e:
        logger.error(f"মডেল ট্রেইনিং এর সময় একটি গুরুতর ত্রুটি ঘটেছে: {e}", exc_info=True)

# --- === প্রধান অংশ (সম্পূর্ণ পরিবর্তন করা হয়েছে) === ---
if __name__ == '__main__':
    # ফোল্ডার তৈরি করা (যদি না থাকে)
    os.makedirs('data', exist_ok=True)
    
    # গুগল ড্রাইভ থেকে বাংলা ডেটাসেট ডাউনলোড করার লিংক
    BENGALI_DATASET_URL = "https://drive.google.com/uc?export=download&id=1GorLcJrmwo-RQmpBaaugdjU1nZayOKBZ"
    BENGALI_DATA_PATH = os.path.join('data', 'Cleaned_Bangla_hatespeech.csv')
    
    # ধাপ ১: ডেটাসেট ডাউনলোড করা
    # যদি ফাইলটি লোকাল মেশিনে না থাকে, তবেই ডাউনলোড করা হবে
    if not os.path.exists(BENGALI_DATA_PATH):
        logger.info(f"লোকাল মেশিনে '{BENGALI_DATA_PATH}' পাওয়া যায়নি। গুগল ড্রাইভ থেকে ডাউনলোড করা হচ্ছে...")
        if not download_file_from_gdrive(BENGALI_DATASET_URL, BENGALI_DATA_PATH):
            logger.error("ডেটাসেট ডাউনলোড ব্যর্থ। ট্রেইনিং বন্ধ করা হচ্ছে।")
            # প্রোগাম থেকে বের হয়ে যাওয়া
            exit()
    else:
        logger.info(f"'{BENGALI_DATA_PATH}' ফাইলটি লোকাল মেশিনে পাওয়া গেছে। ডাউনলোড করার প্রয়োজন নেই।")
    
    # ধাপ ২: মডেল ট্রেইন করা
    # train_bengali_model ফাংশনকে কল করা হচ্ছে
    train_bengali_model(BENGALI_DATA_PATH)
