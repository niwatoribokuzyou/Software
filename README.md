
pip install -r requirements.txt
version.docxにpip listしたやつのっけてる

文字起こし・キャプションBGMプロンプト作成一気にやりたい時

python promptmade.py

文字起こし 

python whisper_test.py

キャプション 

python effb2_test.py

LLM 

python ollama_test.py


sunoAPIとgptはまだうごかない


実装済みの他の生成AIは議事録の一番下に書いてあるからほしかったら言って

# FastAPI Server

## 起動方法

以下のコマンドでFastAPIサーバーを起動します。

```bash
uvicorn main:app --reload
```

## エンドポイント

### `/api/v1/data` (POST)

音声データと環境データを受け取り、音楽生成タスクを開始します。

**リクエストボディ**

```json
{
  "audio_data": "string (base64 encoded)",
  "environmental_data": {
    "key": "value"
  }
}
```

**レスポンス (202 Accepted)**

```json
{
  "message": "Data received and processing started.",
  "task_id": "string (uuid)"
}
```



### `/api/v1/status/{task_id}` (GET)

指定された`task_id`の音楽生成タスクの状況を返します。

**パスパラメータ**

*   `task_id`: `string` - `/api/v1/data`エンドポイントから返されたタスクID。

**レスポンス**

*   **処理中の場合:**
    ```json
    {
      "status": "processing"
    }
    ```
*   **完了した場合:**
    ```json
    {
      "status": "completed",
      "result": "string"
    }
    ```
