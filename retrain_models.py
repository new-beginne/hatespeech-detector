# retrain_models.py

import os
# train_model.py এবং train_model_en.py থেকে ট্রেইনিং ফাংশনগুলো ইম্পোর্ট করা
# (এর জন্য আপনাকে ওই ফাইলগুলোকে মডিউলার করতে হবে, অথবা কোড কপি করতে হবে)
# সহজ করার জন্য, আমরা এখানে ট্রেইনিং ফাংশনগুলোর একটি সরলীকৃত সংস্করণ রাখছি।
import joblib
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import download_file_from_gdrive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BENGALI_DATA_URL = "https://drive.google.com/uc?export=download&id=1GorLcJrmwo-RQmpBaaugdjU1nZayOKBZ"
ENGLISH_DATA_URL = "https://drive.google.com/uc?export=download&id=1UdU6HqyL57ZK9-uU26gkf92zl7a4gL7T"

def get_stop_words(lang='bn'):
    if lang == 'en': return ["a", "an", "the", "in", "is", "are", "i", "you"]
    return ["এবং", "অথবা", "কিন্তু", "যদি", "আমি", "তুমি", "সে", "কি"]

def train_and_save_model(texts, labels, lang_code):
    logger.info(f"'{lang_code}' ভাষার জন্য মডেল রি-ট্রেইন করা হচ্ছে...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words=get_stop_words(lang_code), min_df=3)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', C=10.0, solver='liblinear')
    model.fit(X, labels)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', f'model_{lang_code}.joblib'))
    joblib.dump(vectorizer, os.path.join('models', f'vectorizer_{lang_code}.joblib'))
    logger.info(f"'{lang_code}' ভাষার মডেল সফলভাবে সেভ করা হয়েছে।")

def run_retraining():
    logger.info("===== মডেল রি-ট্রেইনিং প্রক্রিয়া শুরু হচ্ছে =====")
    
    # বাংলা মডেল
    bn_path = os.path.join('data', 'dataset_bn.csv')
    if download_file_from_gdrive(BENGALI_DATA_URL, bn_path):
        df = pd.read_csv(bn_path)
        train_and_save_model(df['texts'].astype(str), df['labels'].astype(int), 'bn')

    # ইংরেজি মডেল
    en_path = os.path.join('data', 'dataset_en.csv')
    if download_file_from_gdrive(ENGLISH_DATA_URL, en_path):
        df = pd.read_csv(en_path)
        train_and_save_model(df['texts'].astype(str), df['labels'].astype(int), 'en')

    logger.info("===== মডেল রি-ট্রেইনিং প্রক্রিয়া সম্পন্ন =====")
    
if __name__ == '__main__':
    run_retraining()
