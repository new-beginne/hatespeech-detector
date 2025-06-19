import os
import re
import joblib
import logging
import time
import datetime
import urllib.parse
from flask import Flask, render_template, request, jsonify
from threading import Lock
from langdetect import detect as detect_language, LangDetectException
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# --- অ্যাপ এবং লগিং কনফিগারেশন ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ক্যাশ বাস্টিং এর জন্য কনটেক্সট প্রসেসর ---
@app.context_processor
def inject_version():
    return {'version': int(time.time())}

# --- ডেটাবেস কানেকশন (Render-এর Environment Variable থেকে নেওয়া হবে) ---
# লোকাল টেস্টিং এর জন্য ডিফল্ট মান হিসেবে আপনার ক্রেডেনশিয়াল ব্যবহার করা হয়েছে
MONGO_USERNAME = os.environ.get("MONGO_USERNAME", "mdhasannirob271")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "Odjlfg1XtZqcjx32")
MONGO_CLUSTER_URL = os.environ.get("MONGO_CLUSTER_URL", "hatespeech.u6hagva.mongodb.net")
DB_NAME = "hatespeech_db"
COLLECTION_NAME = "wrong_predictions"

# পাসওয়ার্ডে বিশেষ চিহ্ন থাকলে সমস্যা এড়ানোর জন্য
encoded_password = urllib.parse.quote_plus(MONGO_PASSWORD)
CONNECTION_STRING = f"mongodb+srv://{MONGO_USERNAME}:{encoded_password}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&appName=hatespeech"

# --- ডেটাবেস ক্লায়েন্ট ---
try:
    client = MongoClient(CONNECTION_STRING)
    client.admin.command('ping')
    db = client[DB_NAME]
    reports_collection = db[COLLECTION_NAME]
    logger.info("✅ MongoDB ডেটাবেসের সাথে সফলভাবে সংযুক্ত হয়েছে।")
except ConnectionFailure as e:
    logger.error(f"❌ MongoDB ডেটাবেসে সংযোগ স্থাপন ব্যর্থ: {e}")
    client = None
except Exception as e:
    logger.error(f"❌ একটি অপ্রত্যাশিত ত্রুটি ঘটেছে: {e}")
    client = None

# --- গ্লোবাল ভ্যারিয়েবল ---
MODELS, VECTORIZERS = {}, {}
wrong_prediction_lock = Lock() # এই লকটি আর ব্যবহৃত হচ্ছে না, তবে রেখে দেওয়া হলো

# --- Helper Functions ---
def load_all_models():
    global MODELS, VECTORIZERS
    languages = {'bn': 'বাংলা', 'en': 'ইংরেজি'}
    logger.info("সকল মডেল লোড করা শুরু হচ্ছে...")
    for lang_code, lang_name in languages.items():
        model_path = os.path.join('models', f'model_{lang_code}.joblib')
        vectorizer_path = os.path.join('models', f'vectorizer_{lang_code}.joblib')
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                MODELS[lang_code] = joblib.load(model_path)
                VECTORIZERS[lang_code] = joblib.load(vectorizer_path)
                logger.info(f"✅ {lang_name} ({lang_code}) মডেল সফলভাবে লোড হয়েছে।")
            except Exception as e:
                logger.error(f"❌ {lang_name} ({lang_code}) মডেল লোড করার সময় ত্রুটি: {e}", exc_info=True)
        else:
            logger.warning(f"⚠️ {lang_name} ({lang_code}) মডেল ফাইল খুঁজে পাওয়া যায়নি।")

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09FFa-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # ... (এই ফাংশনটি আগের মতোই থাকবে, কোনো পরিবর্তন নেই) ...
    if not MODELS or not VECTORIZERS:
        return jsonify({'error': 'কোনো মডেলই সার্ভারে লোড হয়নি। অ্যাডমিনের সাথে যোগাযোগ করুন।'}), 503

    try:
        text = request.get_json().get('text', '').strip()
        if not text:
            return jsonify({'error': 'দয়া করে টেক্সট লিখুন'}), 400

        try:
            lang_code = detect_language(text)
            logger.info(f"শনাক্ত করা ভাষা: '{lang_code}'")
        except LangDetectException:
            logger.warning("ভাষা শনাক্ত করা যায়নি। ডিফল্ট হিসেবে বাংলা (bn) ব্যবহার করা হচ্ছে।")
            lang_code = 'bn'

        model_to_use, vectorizer_to_use = None, None
        if lang_code in MODELS:
            model_to_use = MODELS[lang_code]
            vectorizer_to_use = VECTORIZERS[lang_code]
        elif 'en' in MODELS:
            model_to_use = MODELS['en']
            vectorizer_to_use = VECTORIZERS['en']
        elif 'bn' in MODELS:
            model_to_use = MODELS['bn']
            vectorizer_to_use = VECTORIZERS['bn']

        if model_to_use is None:
            return jsonify({'error': 'প্রেডিকশনের জন্য কোনো উপযুক্ত মডেল পাওয়া যায়নি।'}), 501

        processed_text = preprocess_text(text)
        if not processed_text:
            return jsonify({'text': text, 'result': 'নরমাল স্পিচ', 'hate_prob': 0.0, 'normal_prob': 100.0})

        vectorized_text = vectorizer_to_use.transform([processed_text])
        prediction = model_to_use.predict(vectorized_text)[0]
        probability = model_to_use.predict_proba(vectorized_text)[0]
        
        hate_prob = round(float(probability[1]) * 100, 2)
        normal_prob = round(float(probability[0]) * 100, 2)
        
        result = "হ্যাট স্পিচ" if prediction == 1 else "নরমাল স্পিচ"
        
        return jsonify({
            'text': text,
            'result': result,
            'hate_prob': hate_prob,
            'normal_prob': normal_prob
        })

    except Exception as e:
        logger.error(f"'/detect' রুটে ত্রুটি: {e}", exc_info=True)
        return jsonify({'error': 'সার্ভারে একটি অপ্রত্যাশিত সমস্যা হয়েছে'}), 500

@app.route('/report_wrong', methods=['POST'])
def handle_wrong_report():
    """ব্যবহারকারীর পাঠানো ভুল রিপোর্ট MongoDB-তে সেভ করে।"""
    if client is None:
        return jsonify({'error': 'ডেটাবেস সংযোগে সমস্যা হয়েছে। রিপোর্ট সেভ করা সম্ভব নয়।'}), 500
    try:
        data = request.get_json()
        text, model_prediction = data.get('text'), data.get('model_prediction')
        if not text or model_prediction is None:
            return jsonify({'error': 'অসম্পূর্ণ ডেটা পাঠানো হয়েছে।'}), 400

        correct_label = 1 - int(model_prediction)
        
        report_document = {
            'text': text,
            'correct_label': correct_label,
            'timestamp': datetime.datetime.utcnow()
        }
        # ডেটাবেসে ডকুমেন্ট ইনসার্ট করা
        reports_collection.insert_one(report_document)
        
        logger.info(f"একটি ভুল রিপোর্ট সফলভাবে ডেটাবেসে সেভ করা হয়েছে।")
        return jsonify({'message': 'আপনার মতামতের জন্য ধন্যবাদ!'})
        
    except Exception as e:
        logger.error(f"'/report_wrong' রুটে ডেটাবেসে সেভ করতে ত্রুটি: {e}", exc_info=True)
        return jsonify({'error': 'সার্ভারে রিপোর্ট সেভ করা সম্ভব হয়নি।'}), 500

if __name__ == '__main__':
    # ... (এই অংশটি আগের মতোই থাকবে, কোনো পরিবর্তন নেই) ...
    try:
        import langdetect
        import pymongo
    except ImportError as e:
        logger.error(f"`{e.name}` লাইব্রেরি ইনস্টল করা নেই। `pip install {e.name}` চালান।")
    
    load_all_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
