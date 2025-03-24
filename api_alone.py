import os
import time
import traceback

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from typing import Optional, Union, List
import asyncio
import uvicorn
import aiohttp
from utils.logger import logger
from utils.config import *
from utils.data_manager import MainRequest
from utils.utils import download_audio_from_url,split_stereo_to_mono,merge_and_sort_timestamps,audio_segments_alone,convert_mp3_to_wav
from utils.server import request_vad,request_punc
from utils.SeacoParaformer import SeacoParaformer
logger.info(f"raw audio dir is {raw_audio_dir}")
logger.info(f"segment audio dir is {segment_audio_dir}")

if not os.path.exists(raw_audio_dir):
    os.makedirs(raw_audio_dir)

if not os.path.exists(segment_audio_dir):
    os.makedirs(segment_audio_dir)

if not os.path.exists(asr_model_path):
    logger.info("模型不存在，开始下载模型...")
    #模型下载
    from modelscope import snapshot_download
    model_dir = snapshot_download('iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',cache_dir="ckpt")


asr_model = SeacoParaformer(asr_model_path,quantize=True,intra_op_num_threads=8)

app = FastAPI()

@app.post("/predict")
async def predict(
    sessionId: str = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    hotword: str = Form(...)
):
    audio_file_url = audio_file
    print(f"sessionId:{sessionId},hotword:{hotword}")
    start_time = time.time()
    asr_final_result = []
    try:
        logger.info(f"sessionId is {sessionId}, start loading file:{audio_file_url.filename}")
        # 下载录音到本地
        audio_file_path = os.path.join(raw_audio_dir, audio_file_url.filename)

        with open(audio_file_path, "wb") as file_object:
            file_object.write(await audio_file_url.read())
        if audio_file_path.endswith(".mp3"):
            convert_result = convert_mp3_to_wav(audio_file_path, audio_file_path.replace(".mp3", ".wav"))
            if convert_result:
                os.remove(audio_file_path)
                audio_file_path = audio_file_path.replace(".mp3", ".wav")
        audio_time_stamp = request_vad({"sessionId": sessionId, "file_path": audio_file_path})
        merged_time_stamp = merge_and_sort_timestamps(audio_time_stamp["response"][0], [])
        segments = audio_segments_alone(audio_file_path, merged_time_stamp)
        asr_input = [elem["audio"] for elem in segments]
        logger.info(f"开始调用ASR服务进行推理...")
        try:
            asr_start_time = time.time()
            asr_text = asr_model(asr_input, hotword)
            asr_time = time.time() - asr_start_time
            logger.info(f"asr time is {asr_time}")
            #logger.info(f"sessionId is {sessionId}, asr response is {asr_text}")
        except:
            logger.error({"sessionId": sessionId, "response": "asr识别发生错误", "code": 500, "cost": time.time() - start_time})
            return {"sessionId": sessionId, "response": "asr识别发生错误", "code": 500, "cost": time.time() - start_time}
        logger.info("开始调用标点预测服务...")
        punc_start_time = time.time()

        for i,elem in enumerate(asr_text):
            raw_text = elem["preds"].replace(" ","")
            if raw_text:
                punc_result = request_punc({"sessionId": sessionId, "raw_text": raw_text})
                if punc_result["code"] != 200:
                    punc_result = ""
                else:
                    punc_result = punc_result["response"]
            else:
                continue
            cur_role = segments[i]["spk"]
            asr_final_result.append({"index": i, "speaker": cur_role,"text":punc_result})
        logger.info(f"punc time is {time.time() - punc_start_time}")

        return {"sessionId": sessionId, "response": asr_final_result, "code": 200, "all cost time": time.time() - start_time,"asr time":asr_time}

    except:
        error = traceback.format_exc()
        # logger.info(f"Error, sessionId is {sessionId}, error is {error}")
        return {"sessionId": sessionId, "response": error, "code": 500, "cost": time.time() - start_time}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=18193)

