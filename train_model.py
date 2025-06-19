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

def train_model(csv_file):
    """মডেল ট্রেইন, ইভ্যালুয়েট এবং সেভ করার মূল ফাংশন।"""
    try:
        logger.info("===== মডেল ট্রেইনিং প্রক্রিয়া শুরু হচ্ছে =====")
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
        model_path = os.path.join('models', 'model_bn.joblib')
        vectorizer_path = os.path.join('models', 'vectorizer_bn.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"মডেল সফলভাবে '{model_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info(f"ভেক্টরাইজার সফলভাবে '{vectorizer_path}' ফাইলে সেভ করা হয়েছে।")
        logger.info("===== মডেল ট্রেইনিং প্রক্রিয়া সম্পন্ন =====")
        
    except Exception as e:
        logger.error(f"মডেল ট্রেইনিং এর সময় একটি গুরুতর ত্রুটি ঘটেছে: {e}", exc_info=True)

# --- === প্রধান অংশ (এখানে পরিবর্তন আনা হয়েছে) === ---
if __name__ == '__main__':
    # ফাইলের পাথগুলো নির্ধারণ করা
    RAW_DATA_PATH = os.path.join('data', 'Bangla_hatespeech.csv') # আপনার মূল, অপরিষ্কার ডেটাসেট
    CLEANED_DATA_PATH = os.path.join('data', 'Cleaned_Bangla_hatespeech.csv') # পরিষ্কার করা ডেটাসেট এখানে সেভ হবে

    # ধাপ ১: ডেটা পরিষ্কার করা (যদি প্রয়োজন হয়)
    # যদি পরিষ্কার করা ডেটাসেট না থাকে, কিন্তু মূল ডেটাসেট থাকে, তবে clean_dataset.py চালান
    if not os.path.exists(CLEANED_DATA_PATH) and os.path.exists(RAW_DATA_PATH):
        logger.info(f"'{CLEANED_DATA_PATH}' পাওয়া যায়নি। ডেটা পরিষ্কার করা হচ্ছে...")
        try:
            # clean_dataset.py কে একটি মডিউল হিসেবে ইম্পোর্ট করা হচ্ছে
            import clean_dataset 
            # আপনার clean_dataset.py এর ফাংশনটিকে কল করা হচ্ছে
            clean_dataset.debug_and_clean_dataset(RAW_DATA_PATH, CLEANED_DATA_PATH)
        except ImportError:
            logger.error("'clean_dataset.py' ফাইলটি ইম্পোর্ট করা সম্ভব হয়নি।")
        except Exception as e:
            logger.error(f"ডেটা পরিষ্কার করার সময় ত্রুটি: {e}")

    # ধাপ ২: মডেল ট্রেইন করা
    # নিশ্চিত করুন যে পরিষ্কার করা ডেটাসেটটি বিদ্যমান আছে
    if os.path.exists(CLEANED_DATA_PATH):
        train_model(CLEANED_DATA_PATH)
    else:
        logger.error(f"ট্রেইনিং শুরু করা সম্ভব নয় কারণ '{CLEANED_DATA_PATH}' ফাইলটি পাওয়া যায়নি।")
        logger.error("দয়া করে নিশ্চিত করুন আপনার 'data' ফোল্ডারে একটি ডেটাসেট ফাইল আছে এবং clean_dataset.py সঠিকভাবে চলছে।")