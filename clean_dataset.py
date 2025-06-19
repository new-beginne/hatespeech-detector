import pandas as pd
import re
import os
import logging
from collections import Counter

# --- লগিং কনফিগারেশন ---
# (এই অংশটি train_model.py-এর সাথে সামঞ্জস্যপূর্ণ)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers or len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.FileHandler):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- ধ্রুবক (Constants) ---
HATE_KEYWORDS = {
    "শালা", "পুত", "মাগি", "পোলা", "চুদা", "চোদা", "গালি", "জারজ", "পতিতা", 
    "পুটকি", "আবাল", "খানকি", "কুত্তা", "বাইনচোদ", "মাদারচোদ", "হালা", "বাল",
    "নাস্তিক", "কুথাকার", "হারামি"
}

# --- Helper Functions ---

def clean_text(text):
    """টেক্সট থেকে অপ্রয়োজনীয় অংশ এবং খুব ছোট/বড় বাক্য বাদ দেয়।"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text) # বাংলা অক্ষর বাদে বিশেষ চিহ্ন বাদ
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    words = text.split()
    # খুব ছোট (২ শব্দের কম) বা খুব বড় (৫০০ শব্দের বেশি) বাক্য বাদ দেওয়া
    if len(words) < 2 or len(words) > 500:
        return ""
    return text

def apply_rules(df):
    """
    নিয়ম-ভিত্তিক লজিক প্রয়োগ করে লেবেল সংশোধন করে।
    """
    logger.info("নিয়ম-ভিত্তিক লেবেল সংশোধন করা হচ্ছে...")
    
    def rule_based_label(row):
        text = row['texts']
        label = row['labels']
        # যদি কোনো হ্যাট কিওয়ার্ড থাকে, তাহলে লেবেল জোর করে 1 করা হবে
        if any(keyword in text for keyword in HATE_KEYWORDS):
            return 1
        return label

    df['labels'] = df.apply(rule_based_label, axis=1)
    return df

def remove_inconsistent_duplicates(df):
    """
    ডুপ্লিকেট টেক্সটগুলোর জন্য ভোটিং পদ্ধতির মাধ্যমে একটি লেবেল নির্বাচন করে।
    """
    logger.info("অসামঞ্জস্যপূর্ণ ডুপ্লিকেট সরানো হচ্ছে...")
    # একই টেক্সটের জন্য সবচেয়ে কমন লেবেলটি নেওয়া হচ্ছে
    agg_df = df.groupby('texts').agg(
        labels=('labels', lambda x: Counter(x).most_common(1)[0][0])
    ).reset_index()
    return agg_df

# --- প্রধান ফাংশন ---

def debug_and_clean_dataset(input_file, output_file):
    """
    মূল ডেটাসেট ফাইলটিকে পড়ে, পরিষ্কার করে এবং মডেলের জন্য উপযুক্ত ফরম্যাটে সেভ করে।
    """
    logger.info("===== ডেটা ক্লিনিং প্রক্রিয়া শুরু হচ্ছে =====")
    try:
        # ডেটাসেট লোড করা
        df = pd.read_csv(input_file)
        logger.info(f"'{input_file}' থেকে {len(df)} টি লাইন লোড করা হয়েছে।")
        
        # কলামের নাম ঠিক করা
        if 'text' in df.columns and 'texts' not in df.columns:
            df.rename(columns={'text': 'texts', 'label': 'labels'}, inplace=True)

        # প্রাথমিক ক্লিনিং
        df.dropna(subset=['texts', 'labels'], inplace=True)
        df['labels'] = pd.to_numeric(df['labels'], errors='coerce')
        df.dropna(subset=['labels'], inplace=True)
        df['labels'] = df['labels'].astype(int)
        df = df[df['labels'].isin([0, 1])]
        
        # টেক্সট পরিষ্কার করা
        df['texts'] = df['texts'].apply(clean_text)
        df.dropna(subset=['texts'], inplace=True)
        df = df[df['texts'] != '']
        logger.info(f"প্রাথমিক টেক্সট ক্লিনিং এর পর {len(df)} টি লাইন অবশিষ্ট আছে।")

        # নিয়ম-ভিত্তিক লেবেল সংশোধন
        df = apply_rules(df)

        # ডুপ্লিকেট সরানো
        original_count = len(df)
        df = remove_inconsistent_duplicates(df)
        logger.info(f"{original_count - len(df)} টি ডুপ্লিকেট লাইন সরানো হয়েছে।")
        
        logger.info(f"চূড়ান্ত পরিষ্কার ডেটাসেটে {len(df)} টি লাইন আছে।")

        # চূড়ান্ত ফাইল সেভ করা
        if len(df) > 0:
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"পরিষ্কার ডেটাসেট সফলভাবে '{output_file}' ফাইলে সেভ করা হয়েছে।")
        else:
            logger.error("ক্লিনিং এর পর কোনো ডেটা অবশিষ্ট নেই। আউটপুট ফাইল তৈরি করা হয়নি।")

        logger.info("===== ডেটা ক্লিনিং প্রক্রিয়া সম্পন্ন =====")

    except FileNotFoundError:
        logger.error(f"ত্রুটি: ইনপুট ফাইল '{input_file}' খুঁজে পাওয়া যায়নি।")
    except Exception as e:
        logger.error(f"ডেটা ক্লিনিং এর সময় একটি মারাত্মক ত্রুটি ঘটেছে: {e}", exc_info=True)

# --- স্ক্রিপ্ট চালনার অংশ ---
if __name__ == '__main__':
    # ফোল্ডার তৈরি করা (যদি না থাকে)
    os.makedirs('data', exist_ok=True)
    
    # ফাইলের পাথ নির্ধারণ করা
    input_csv_path = os.path.join('data', 'Bangla_hatespeech.csv')
    output_csv_path = os.path.join('data', 'Cleaned_Bangla_hatespeech.csv')
    
    # প্রধান ফাংশনটি চালানো
    debug_and_clean_dataset(input_csv_path, output_csv_path)