import requests

def download_file_from_gdrive(file_id: str, destination: str):
    """
    Google Drive থেকে ফাইল ডাউনলোড করে লোকাল ফাইলে সংরক্ষণ করে।
    file_id: ড্রাইভের ফাইল ID
    destination: কোন নামে বা কোথায় ফাইলটা সেভ হবে
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """
    ড্রাইভ ফাইল বড় হলে confirmation token দরকার হয়, এটা সেটা বের করে
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    """
    রেসপন্স থেকে ডেটা কেটে কেটে ফাইলে লেখে
    """
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # ফাঁকা না হলে লেখো
                f.write(chunk)
