from fastapi import FastAPI, BackgroundTasks, HTTPException 
from pydantic import BaseModel
import uuid
import base64

app = FastAPI()

class DataPayload(BaseModel):
	audio_data: str
	enviromental_data: dict

task_status_db = {}

@app.get("/api/v1/data", status_code=202)
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
		payload.enviromental_data
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

