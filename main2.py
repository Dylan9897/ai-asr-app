import os
import time
import traceback
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
import aiohttp
from utils.logger import logger
from utils.config import *
from utils.data_manager import MainRequest
from utils.utils import download_audio_from_url, split_stereo_to_mono, merge_and_sort_timestamps, audio_segments, \
    convert_mp3_to_wav
from utils.server import request_vad, request_punc
from utils.SeacoParaformer import SeacoParaformer
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

asr_model = SeacoParaformer(asr_model_path,quantize=True,intra_op_num_threads=8)

app = FastAPI()

# 配置并发处理参数
MAX_CONCURRENT_REQUESTS = 10  # 最大并发请求数
MAX_QUEUE_SIZE = 1000  # 最大队列大小
TIMEOUT = 300  # 请求超时时间（秒）

# 创建有限的信号量来控制并发
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# 创建线程池处理CPU密集型任务
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

# 创建请求追踪字典
request_tracker: Dict[str, dict] = {}


class RequestStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


async def process_request(request: MainRequest) -> dict:
    """处理单个请求的核心逻辑"""
    sessionId = request.sessionId
    start_time = time.time()

    try:
        # 更新请求状态
        request_tracker[sessionId] = {
            "status": RequestStatus.PROCESSING,
            "start_time": start_time,
            "progress": 0
        }

        # 下载音频文件
        async with aiohttp.ClientSession() as session:
            tasks = download_audio_from_url(session, request.audio_file_url, raw_audio_dir)
            result = await asyncio.gather(tasks)
            request_tracker[sessionId]["progress"] = 20

        if not result[0][1]:
            raise Exception("音频文件下载失败")

        audio_file_path = result[0][0]

        # 音频预处理
        # 将CPU密集型任务放入线程池
        def audio_preprocessing():
            if audio_file_path.endswith(".mp3"):
                wav_path = audio_file_path.replace(".mp3", ".wav")
                convert_mp3_to_wav(audio_file_path, wav_path)
                return wav_path
            return audio_file_path

        processed_audio_path = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            audio_preprocessing
        )
        request_tracker[sessionId]["progress"] = 40

        # 音频分段处理
        segments = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: audio_segments(processed_audio_path, processed_audio_path, [])  # 简化示例
        )
        request_tracker[sessionId]["progress"] = 60

        # ASR处理
        asr_input = [elem["audio"] for elem in segments]
        asr_results = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: asr_model(asr_input, request.hotword)
        )
        request_tracker[sessionId]["progress"] = 80

        # 结果处理和返回
        final_results = []
        for i, result in enumerate(asr_results):
            processed_text = await process_text(result, sessionId)
            final_results.append({
                "index": i,
                "speaker": segments[i]["spk"],
                "text": processed_text
            })

        request_tracker[sessionId]["status"] = RequestStatus.COMPLETED
        request_tracker[sessionId]["progress"] = 100

        return {
            "sessionId": sessionId,
            "response": final_results,
            "code": 200,
            "cost": time.time() - start_time
        }

    except Exception as e:
        request_tracker[sessionId]["status"] = RequestStatus.FAILED
        logger.error(f"Error processing request {sessionId}: {str(e)}")
        return {
            "sessionId": sessionId,
            "response": str(e),
            "code": 500,
            "cost": time.time() - start_time
        }


async def process_text(text_result, sessionId):
    """处理文本的异步函数"""
    try:
        raw_text = text_result["preds"].replace(" ", "")
        if not raw_text:
            return ""

        punc_result = await request_punc({"sessionId": sessionId, "raw_text": raw_text})
        return punc_result["response"] if punc_result["code"] == 200 else raw_text
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return raw_text


@app.post("/predict")
async def predict(request: MainRequest, background_tasks: BackgroundTasks):
    """API入口点"""
    sessionId = request.sessionId

    # 检查是否已经存在相同的请求
    if sessionId in request_tracker:
        return {
            "sessionId": sessionId,
            "status": request_tracker[sessionId]["status"],
            "progress": request_tracker[sessionId]["progress"]
        }

    # 使用信号量控制并发
    async with semaphore:
        return await process_request(request)


@app.get("/status/{sessionId}")
async def get_status(sessionId: str):
    """获取请求处理状态"""
    if sessionId in request_tracker:
        return request_tracker[sessionId]
    return {"status": "not_found"}


# 清理过期请求记录的后台任务
async def cleanup_old_requests():
    while True:
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, data in request_tracker.items()
            if current_time - data["start_time"] > TIMEOUT
        ]
        for session_id in expired_sessions:
            del request_tracker[session_id]
        await asyncio.sleep(60)  # 每分钟检查一次


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    asyncio.create_task(cleanup_old_requests())


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=18002, workers=4)