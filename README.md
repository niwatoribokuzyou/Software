# README

これは、音声と環境データを受信し、音楽を生成し、ステータスを確認して生成された音楽を取得するためのエンドポイントを提供するFastAPIアプリケーションです。

## セットアップ

アプリケーション実行までのセットアップ手順です。  
以下の実行は、**リポジトリ直下**で行うことを前提とします。

## インストール

1. リポジトリをクローンします。
2. pipを使用して必要なPythonパッケージをインストールします。
3. .envファイルを作成
4. ローカルモデルのインストールは初回実行時に自動的に行われます。

依存関係をインストール
```bash
pip install -r requirements.txt
```
.env
```env
OPEN_AI_APIKEY = ここにAPI KEY
WEATHER_APIKEY = ここにAPI KEY
```

## アプリケーションの実行

uvicornを使用してアプリケーションを実行できます。

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## エンドポイント

### `/api/v1/data` (POST)

音声と環境データを受信し、音楽生成プロセスを開始します。

**リクエストボディ**

```json
{
  "audio_data": "string (base64 encoded)",
  "environmental_data": {
    "temperature": "float",
    "pressure": "float",
    "humidity": "float",
    "lux": "float"
  }
}
```

**レスポンス (202 Accepted)**

```json
{
  "message": "データを受信し、処理を開始しました。",
  "task_id": "string (uuid)"
}
```

### `/api/v1/status/{task_id}` (GET)

音楽生成タスクのステータスを返します。

**パスパラメータ**

*   `task_id`: `string` - `/api/v1/data`エンドポイントから返されたタスクID。

**レスポンス**

*   **処理中:**
    ```json
    {
      "status": "processing"
    }
    ```
*   **完了:**
    ```json
    {
      "status": "completed",
      "result": "string (base64 encoded audio)",
      "min_color": "string (#RRGGBB)",
      "max_color": "string (#RRGGBB)",
      "bpm": "float"
    }
    ```

### `/api/v1/task_list` (GET)

複数のタスクのステータスを確認します。少なくとも1つのタスクが完了している場合は`true`を、それ以外の場合は`false`を返します。

**クエリパラメータ**

*   `task_ids`: `list[str]` - 確認するタスクIDのリスト。

**レスポンス**

```json
true
```
または
```json
false
```

### `/api/v1/get_mock_data` (GET)

テスト目的で、事前に生成されたモックオーディオファイルを返します。

**レスポンス**

```json
{
  "status": "completed",
  "result": "string (base64 encoded audio)",
  "min_color": "#EC0101F8",
  "max_color": "#D3E0E9",
  "bpm": 120
}
```
