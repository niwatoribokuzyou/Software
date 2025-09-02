import requests

url = "http://localhost:8000/api/v1/task_list"

# 複数の task_id を送る
payload = {
    "task_ids": ["task1", "task2", "task3"]
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # HTTPエラーがあれば例外を出す
    data = response.json()
    print(data)  # {"completed": True} または {"completed": False}
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error: {http_err}, {response.text}")
except Exception as err:
    print(f"Other error: {err}")
