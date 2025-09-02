from fastapi import FastAPI, BackgroundTasks, HTTPException 
from pydantic import BaseModel
import uvicorn
import uuid
import base64
from gpt_test import chat_with_gpt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio
from music_generater import generate_music
app = FastAPI()

class DataPayload(BaseModel):
    audio_data: str
    environmental_data: dict

task_status_db = {}

def generate_music_task(task_id: str, audio_data: str, env_data: dict):
	"""
  時間のかかる音楽生成処理をシミュレートするバックグラウンドタスク
  """
	print(f"Task {task_id}: Processing started...")

  # ここで音楽生成のロジックを実行
  # 例: audio_dataをデコードし、env_dataと組み合わせて音楽を生成	
	decoded_audio = base64.b64decode(audio_data)

	temperature = env_data.get("temperature", 25)
	pressure = env_data.get("pressure", 1013)
	humidity = env_data.get("humidity", 50)
	illuminance = env_data.get("lux", 500)


	# ここにサンプリングの処理を実装
	caption = generate_audio_caption(decoded_audio)

	stt_data = transcribe_audio(decoded_audio)

	prompt, min_color, max_color = chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance)
	print("Generated Prompt:", prompt)

	# sunoを実装できたらここ
	music = generate_music(prompt, decoded_audio)

  # 処理が完了したら、データベースの状態を更新
	task_status_db[task_id] = {
    "status": "completed",
    "result": music,
    "min_color": min_color,
    "max_color": max_color,
  }
	print(f"Task {task_id}: Processing completed.")

@app.post("/api/v1/data", status_code=202)
async def receive_data(payload: DataPayload, background_tasks: BackgroundTasks):
	"""
	ラズパイから音声データと環境データを受付、音楽生成を開始
	"""
	
	task_id = str(uuid.uuid4())

	task_status_db[task_id] = {"status": "processing"}

	background_tasks.add_task(
		generate_music_task,
		task_id,
		payload.audio_data,
		payload.environmental_data
	)

	return {"message": "Data received and processing started.", "task_id": task_id} 

@app.get("/api/v1/status/{task_id}")
async def get_task_status(task_id: str):
	"""
	指定されたtask_idの処理状況を返す
	"""
	
	if task_id not in task_status_db:
		raise HTTPException(status_code=404, detail="Task ID not found")

	task = task_status_db[task_id]

	if task["status"] == "completed":
		return {"status": "completed", "result": task["result"]}
	else:
		return {"status": "processing"}

@app.get("/api/v1/get_mock_data")
async def get_mock_data():
	"""
	生成済みの音源を返す
	Returns:
		dict: A JSON object with the following structure:
      {
				"status": "completed",
				"result": "<base64-encoded audio string>"
			}
	Error Cases:
		- If the audio file does not exist or cannot be read, an HTTP 500 error may be returned.
	"""
	sound_path = "output-f.mp3"
	try:
		with open(sound_path, "rb") as f:
			binary_mp3 = f.read()
	except FileNotFoundError:
		raise HTTPException(status_code=404, detail="Mock audio file not found.")

	encoded_audio = base64.b64encode(binary_mp3).decode("utf-8")
	return {"status": "completed", "result": encoded_audio, "min_color": "#A8BFCF", "max_color": "#D3E0E9"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
