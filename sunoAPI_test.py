

# import requests
# import time

# # ----------------------------
# # 1. 設定
# # ----------------------------
# BASE_URL = 'https://api.sunoapi.org/api/v1'
# API_KEY = '2ea0a6bf4e18833e4e78f12c2fa4fa7c'  # 自分のSuno APIキーに置き換える

# HEADERS = {
#     'Authorization': f'Bearer {API_KEY}',
#     'Content-Type': 'application/json'
# }

# # PROMPT_FILE = 'prompt.txt'      # 曲生成用プロンプトを記載したファイル
# OUTPUT_FILE = 'generated_music.mp3'
# PROMPT_FILE = './answer_data/output_bgm_prompt.txt'      # ここに曲のプロンプトを記載したテキストファイル

# # ----------------------------
# # 2. プロンプト読み込み
# # ----------------------------
# try:
#     with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
#         prompt = f.read().strip()
#         if not prompt:
#             raise ValueError("プロンプトが空です。")
# except Exception as e:
#     print(f"プロンプトの読み込みエラー: {e}")
#     exit()

# print(f"読み込んだプロンプト:\n{prompt}\n")

# # ----------------------------
# # 3. 曲生成リクエスト
# # ----------------------------
# data = {
#     'prompt': prompt,
#     'model': 'V4',
#     'customMode': False,
#     'instrumental': False,
#     'callBackUrl': 'https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox'  # ダミーURLでOK
# }

# print("曲生成リクエスト送信中...")
# response = requests.post(f'{BASE_URL}/generate', headers=HEADERS, json=data)
# response.raise_for_status()
# result = response.json()

# if result.get('code') != 200:
#     print("曲生成リクエストに失敗しました:", result.get('msg'))
#     exit()

# task_id = result['data']['taskId']
# print("result", result)
# exit()
# status_url = result['data']['status_url']  # 生成APIが返すtask専用ステータスURL
# print(f"曲生成タスクID: {task_id}")

# # ----------------------------
# # 4. 曲生成完了までポーリング
# # ----------------------------
# while True:
#     status_response = requests.get(status_url, headers=HEADERS)
#     status_response.raise_for_status()
#     status_data = status_response.json()['data']
#     status = status_data['status']
    
#     if status == 'completed':
#         print("曲生成完了！")
#         audio_url = status_data['audio_url']
#         break
#     elif status == 'failed':
#         print("曲生成失敗")
#         exit()
#     else:
#         print("生成中...")
#         time.sleep(5)  # 5秒ごとに確認

# # ----------------------------
# # 5. 曲をダウンロードして保存
# # ----------------------------
# print("曲をダウンロード中...")
# with requests.get(audio_url, stream=True) as r:
#     r.raise_for_status()
#     with open(OUTPUT_FILE, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=1024):
#             f.write(chunk)

# print(f"曲を保存しました: {OUTPUT_FILE}")



import requests
import time

# ----------------------------
# 1. 設定
# ----------------------------
BASE_URL = 'https://api.sunoapi.org/api/v1'
API_KEY = '4015bb42c039ff70bfa42ed950f05806'  # 自分のSuno APIキーに置き換える

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

PROMPT_FILE = './answer_data/output_bgm_prompt.txt'  # 曲生成用プロンプト
OUTPUT_FILE = 'generated_music.mp3'

# ----------------------------
# 2. プロンプト読み込み
# ----------------------------
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
        if not prompt:
            raise ValueError("プロンプトが空です。")
except Exception as e:
    print(f"プロンプトの読み込みエラー: {e}")
    exit()

print(f"読み込んだプロンプト:\n{prompt}\n")

# ----------------------------
# 3. 曲生成リクエスト
# ----------------------------
data = {
    'prompt': prompt,
    'model': 'V4',
    'customMode': False,
    'instrumental': False,
    'callBackUrl': 'https://example.com/callback'  # ダミーURL
}

print("曲生成リクエスト送信中...")
response = requests.post(f'{BASE_URL}/generate', headers=HEADERS, json=data)
response.raise_for_status()
result = response.json()

if result.get('code') != 200:
    print("曲生成リクエストに失敗しました:", result.get('msg'))
    exit()

task_id = result['data']['taskId']
status_url = f'{BASE_URL}/generate/{task_id}/status'  # taskIdからステータスURLを生成
print(f"曲生成タスクID: {task_id}")

# ----------------------------
# 4. 曲生成完了までポーリング
# ----------------------------
while True:
    status_response = requests.get(status_url, headers=HEADERS)
    status_response.raise_for_status()
    status_data = status_response.json()['data']
    status = status_data['status']
    
    if status == 'completed':
        print("曲生成完了！")
        audio_url = status_data['audio_url']
        break
    elif status == 'failed':
        print("曲生成失敗")
        exit()
    else:
        print("生成中...")
        time.sleep(5)  # 5秒ごとに確認

# ----------------------------
# 5. 曲をダウンロードして保存
# ----------------------------
print("曲をダウンロード中...")
with requests.get(audio_url, stream=True) as r:
    r.raise_for_status()
    with open(OUTPUT_FILE, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)

print(f"曲を保存しました: {OUTPUT_FILE}")
