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

# --- লগিং কনফিগারেশন ---
# একটি লগ ফাইল এবং কনসোল উভয় জায়গায় লগ দেখানোর জন্য
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='training.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
# কনসোলে আউটপুট দেখানোর জন্য হ্যান্ডলার
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
                # আপনার কলামের নাম 'text', 'label' অথবা 'texts', 'labels' দুটোই হ্যান্ডেল করবে
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
    # অফেন্সিভ বা গুরুত্বপূর্ণ শব্দ এখানে যোগ করা যাবে না।
    stop_words = {
        "এবং", "অথবা", "কিন্তু", "যদি", "তবে", "আমি", "তুমি", "সে", "আমরা", "তোমরা", 
        "তারা", "এই", "যে", "কি", "না", "হ্যাঁ", "তো", "কে", "আর", "ও", "নেই", "হবে", 
        "হয়", "করতে", "করবে", "যাও", "গেলে", "থেকে", "পর্যন্ত", "চাই", "করা", "তাহলে",
        "করে", "হল", "হলে", "যেমন", "তেমন", "সব", "এক", "দুই", "করেছে", "নয়", "হত",
        "তা", "তাই", "ওই", "এটা", "সেটা", "এখন", "আগে", "পরে", "মধ্যে"
    }
    return list(stop_words)

def train_model(csv_file):
    """মডেল ট্রেইন, ইভ্যালুয়েট এবং সেভ করার মূল ফাংশন।"""
    try:
        logger.info("===== মডেল ট্রেইনিং প্রক্রিয়া শুরু হচ্ছে =====")
        texts, labels = prepare_dataset(csv_file)
        
        if not texts or len(texts) < 100:
            logger.error("ট্রেইনিং বন্ধ করা হচ্ছে। ডেটাসেটে পর্যাপ্ত ডেটা নেই (< 100)।")
            return

        # টেক্সট প্রি-প্রসেসিং এর কোনো ধাপ এখানে নেই, কারণ TfidfVectorizer নিজেই lowercase করে
        # এবং re ব্যবহার করে প্রি-প্রসেসিং করলে কিছু গুরুত্বপূর্ণ তথ্য হারিয়ে যেতে পারে।
        # Vectorizer-এর `token_pattern` অনেক কাজ করে দেয়।
        
        # --- ভেক্টরাইজেশন (সবচেয়ে গুরুত্বপূর্ণ পরিবর্তন) ---
        logger.info("ফিচার ভেক্টরাইজেশন শুরু হচ্ছে...")
        vectorizer = TfidfVectorizer(
            max_features=15000,       # <-- ফিচার সংখ্যা বাড়ানো হয়েছে
            ngram_range=(1, 3),       # <-- শব্দ, জোড়া শব্দ এবং তিনটি শব্দের কম্বিনেশন
            stop_words=get_stop_words(), # <-- অফেন্সিভ শব্দ ছাড়া শুধুমাত্র সাধারণ স্টপ ওয়ার্ড
            min_df=3,                 # <-- ফিচার হতে হলে শব্দটি কমপক্ষে ৩ বার আসতে হবে
            max_df=0.9,               # <-- ৯০% এর বেশি ডকুমেন্টে থাকা শব্দ বাদ যাবে
            token_pattern=r'(?u)\b\w+\b' # <-- শুধুমাত্র শব্দ (চিহ্ন নয়) টোকেন হিসেবে গণ্য হবে
        )
        X = vectorizer.fit_transform(texts)
        logger.info(f"ভেক্টরাইজড ডেটা আকার: {X.shape}")

        # ডেটা স্প্লিট করা (ট্রেনিং এবং টেস্টিং এর জন্য)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # --- মডেল কনফিগারেশন এবং ট্রেনিং ---
        logger.info("মডেল ট্রেইনিং শুরু হচ্ছে...")
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=10.0,                   # <-- C এর মান বাড়ানো হয়েছে আরও ভালোভাবে শেখার জন্য
            solver='liblinear',
            penalty='l2',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # --- ইভ্যালুয়েশন ---
        logger.info("===== মডেল ইভ্যালুয়েশন =====")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Hate'])
        
        logger.info(f"মডেলের অ্যাকুরেসি: {accuracy:.2%}")
        logger.info("ক্লাসিফিকেশন রিপোর্ট:\n" + report)
        
        # --- মডেল এবং ভেক্টরাইজার সংরক্ষণ ---
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'model.joblib')
        vectorizer_path = os.path.join('models', 'vectorizer.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"মডেল সফলভাবে '{model_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info(f"ভেক্টরাইজার সফলভাবে '{vectorizer_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info("===== মডেল ট্রেইনিং প্রক্রিয়া সম্পন্ন =====")
        
    except Exception as e:
        logger.error(f"মডেল ট্রেইনিং এর সময় একটি গুরুতর ত্রুটি ঘটেছে: {e}", exc_info=True)

if __name__ == '__main__':
    # আপনার পরিষ্কার করা ডেটাসেট ফাইলের পাথ
    cleaned_csv_path = os.path.join('data', 'Cleaned_Bangla_hatespeech.csv')
    
    # মডেল ট্রেইনিং ফাংশন কল করা
    train_model(cleaned_csv_path)