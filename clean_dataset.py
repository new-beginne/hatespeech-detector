import csv
import re
import os
from collections import Counter

HATE_KEYWORDS = {
    "শালা", "পুত", "মাগি", "পোলা", "চুদা", "চোদা", "গালি", "জারজ", "পতিতা", 
    "পুটকি", "আবাল", "খানকি", "কুত্তা", "বাইনচোদ", "মাদারচোদ", "হালা", "বাল",
    "নাস্তিক", "কুথাকার", "হারামি"
}

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) < 2:
        return ""
    if len(words) > 500:
        text = ' '.join(words[:500])
    return text.lower()

def detect_hate_keywords(text):
    return 1 if any(keyword in text for keyword in HATE_KEYWORDS) else 0

def clean_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"ত্রুটি: ইনপুট ফাইল '{input_file}' খুঁজে পাওয়া যায়নি।")
        return

    cleaned_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = 0
        valid_rows = 0
        for row in reader:
            total_rows += 1
            if 'texts' in row and 'labels' in row and row['texts']:
                try:
                    label = int(row['labels'])
                    if label not in [0, 1]:
                        print(f"ভুল লেবেল সরানো হচ্ছে: {row}")
                        continue
                    text = clean_text(row['texts'])
                    if not text:
                        continue
                    if detect_hate_keywords(text) == 1:
                        label = 1
                    valid_rows += 1
                    cleaned_data.append({'texts': text, 'labels': str(label)})
                except ValueError:
                    print(f"লেবেল ইন্টে কনভার্ট ব্যর্থ: {row}")
        print(f"মোট রো: {total_rows}, বৈধ রো: {valid_rows}")

    if cleaned_data:
        df = pd.DataFrame(cleaned_data)
        df = df.groupby('texts').agg({'labels': lambda x: Counter(x).most_common()[0][0]}).reset_index()
        print(f"ডুপ্লিকেট রিমুভের পর রো: {len(df)}")
    else:
        print("ত্রুটি: কোনো ডেটা ক্লিন করা যায়নি।")
        return

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["texts", "labels"])
        writer.writeheader()
        writer.writerows(df.to_dict('records'))

    print(f"পরিষ্কার ডেটাসেট সফলভাবে '{output_file}' ফাইলে সংরক্ষিত হয়েছে।")

if __name__ == '__main__':
    import pandas as pd
    os.makedirs('data', exist_ok=True)
    input_csv_path = 'data/Bangla_hatespeech.csv'
    output_csv_path = 'data/Cleaned_Bangla_hatespeech.csv'
    clean_dataset(input_csv_path, output_csv_path)