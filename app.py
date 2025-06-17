import os
import re
import csv
import joblib
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HATE_KEYWORDS = {
    "শালা", "পুত", "মাগি", "পোলা", "চুদা", "চোদা", "গালি", "জারজ", "পতিতা", 
    "পুটকি", "আবাল", "খানকি", "কুত্তা", "বাইনচোদ", "মাদারচোদ", "হালা", "বাল"
}

def prepare_dataset():
    texts, labels = [], []
    csv_file = 'data/Cleaned_Bangla_hatespeech.csv'
    if not os.path.exists(csv_file):
        logger.error(f"'{csv_file}' ফাইলটি খুঁজে পাওয়া যায়নি! ফালব্যাক ডেটা ব্যবহার করা হচ্ছে।")
        return get_fallback_data()
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'text' in row and 'label' in row and row['text'] and row['label'] in ['0', '1']:
                    texts.append(row['text'])
                    labels.append(int(row['label']))
        
        if not texts: return get_fallback_data()
        
        total, hate_count = len(texts), sum(labels)
        logger.info(f"ডেটাসেট পরিসংখ্যান: মোট={total}, হ্যাট={hate_count} ({(hate_count/total):.1%}), নরমাল={total-hate_count} ({(total-hate_count)/total:.1%})")
        return texts, labels
    except Exception as e:
        logger.error(f"ডেটাসেট লোড করতে ত্রুটি: {e}")
        return get_fallback_data()

def get_fallback_data():
    logger.warning("ফালব্যাক ডেটাসেট ব্যবহার হচ্ছে...")
    return [
        "তুমি খুব বোকা মানুষ", "আমি তোমাকে ঘৃণা করি", "তুমি আমার সেরা বন্ধু", "আমি তোমার প্রতি কৃতজ্ঞ",
        "তোমার মত অপদার্থ আর দেখিনি", "তুমি সত্যিই অসাধারণ", "তুমি সম্পূর্ণ ব্যর্থ", "তোমার সাহায্যের জন্য ধন্যবাদ"
    ], [1, 1, 0, 0, 1, 0, 1, 0]

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09FFa-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_stop_words():
    return [
        "এবং", "অথবা", "কিন্তু", "যদি", "তবে", "যে", "কি", "না", "হ্যাঁ", "তো", "কে", "আর", "ও", 
        "নেই", "হবে", "হয়", "করতে", "করবে", "যাও", "গেলে", "থেকে", "পর্যন্ত", "তাহলে", "করে", "হল", "হলে"
    ]

def train_model():
    try:
        logger.info("মডেল ট্রেইনিং শুরু হচ্ছে...")
        texts, labels = prepare_dataset()
        if len(texts) < 20:
            logger.error("ট্রেইনিং এর জন্য পর্যাপ্ত ডেটা নেই।")
            return None, None
            
        processed_texts = [preprocess_text(text) for text in texts]
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=get_stop_words())
        X = vectorizer.fit_transform(processed_texts)
        
        model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='liblinear')
        
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info("ক্লাসিফিকেশন রিপোর্ট:\n" + report)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.joblib')
        joblib.dump(vectorizer, 'models/vectorizer.joblib')
        
        logger.info("মডেল সফলভাবে ট্রেইন ও সেভ করা হয়েছে।")
        return model, vectorizer
    except Exception as e:
        logger.error(f"মডেল ট্রেইনিং এ মারাত্মক ত্রুটি: {e}", exc_info=True)
        return None, None

def load_model():
    model_path, vectorizer_path = 'models/model.joblib', 'models/vectorizer.joblib'
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return train_model()
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        logger.error(f"মডেল লোড করতে সমস্যা ({e}), পুনরায় ট্রেইন করা হচ্ছে...")
        return train_model()

def detect_hate_speech(text):
    model, vectorizer = load_model()
    if model is None: return 0, [1.0, 0.0]

    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    contains_hate_keyword = any(keyword in processed_text for keyword in HATE_KEYWORDS)
    if contains_hate_keyword:
        prediction = 1
        
    return prediction, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    text = request.get_json().get('text', '').strip()
    if not text: return jsonify({'error': 'দয়া করে টেক্সট লিখুন'}), 400
    
    prediction, probability = detect_hate_speech(text)
    hate_prob = round(probability[1] * 100, 2)
    normal_prob = round(probability[0] * 100, 2)
    
    result = "হ্যাট স্পিচ" if prediction == 1 else "নরমাল স্পিচ"
    return jsonify({'result': result, 'hate_prob': hate_prob, 'normal_prob': normal_prob})

if __name__ == '__main__':
    if not os.path.exists('data/Cleaned_Bangla_hatespeech.csv') and os.path.exists('data/Bangla_hatespeech.csv'):
        try:
            import clean_dataset
            clean_dataset.clean_dataset('data/Bangla_hatespeech.csv', 'data/Cleaned_Bangla_hatespeech.csv')
        except ImportError:
            logger.error("'clean_dataset.py' পাওয়া যায়নি।")

    load_model()
    app.run(host='0.0.0.0', port=5000)