import os
import re
import csv
import joblib
import logging
import time # <-- নতুন ইম্পোর্ট
from flask import Flask, render_template, request, jsonify
from threading import Lock
from langdetect import detect as detect_language, LangDetectException

# --- অ্যাপ এবং লগিং কনফিগারেশন ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- === নতুন কোড ব্লক শুরু === ---
@app.context_processor
def inject_version():
    """
    CSS এবং JS ফাইলের জন্য ক্যাশ বাস্টিং ভার্সন তৈরি করে।
    এটি নিশ্চিত করবে যে ব্রাউজার সবসময় নতুন ফাইল লোড করে।
    """
    return {'version': int(time.time())}
# --- === নতুন কোড ব্লক শেষ === ---

# --- গ্লোবাল ভ্যারিয়েবল ---
MODELS = {}
VECTORIZERS = {}
wrong_prediction_lock = Lock()

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
    try:
        data = request.get_json()
        text, model_prediction = data.get('text'), data.get('model_prediction')
        if not text or model_prediction is None:
            return jsonify({'error': 'অসম্পূর্ণ ডেটা পাঠানো হয়েছে।'}), 400

        correct_label = 1 - int(model_prediction)
        report_file = os.path.join('data', 'wrong_predictions.csv')
        os.makedirs('data', exist_ok=True)
        
        with wrong_prediction_lock:
            file_exists = os.path.isfile(report_file)
            with open(report_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['text', 'correct_label'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'text': text, 'correct_label': correct_label})
        
        logger.info(f"একটি ভুল রিপোর্ট সফলভাবে সেভ করা হয়েছে।")
        return jsonify({'message': 'আপনার মতামতের জন্য ধন্যবাদ!'})
        
    except Exception as e:
        logger.error(f"'/report_wrong' রুটে ত্রুটি: {e}", exc_info=True)
        return jsonify({'error': 'সার্ভারে রিপোর্ট সেভ করা সম্ভব হয়নি।'}), 500

if __name__ == '__main__':
    try:
        import langdetect
    except ImportError:
        logger.error("`langdetect` লাইব্রেরি ইনস্টল করা নেই। `pip install langdetect` চালান।")
    
    load_all_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
