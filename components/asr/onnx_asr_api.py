# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 14:26

from fastapi import FastAPI,Request


import uvicorn
import numpy as np
import time
import os
import sys
sys.path.append(os.getcwd())


from utils.logger import logger
from utils.data_manager import AsrRequest,AsrResponseModel
from utils.config import asr_model_path

from components.asr.model import SeacoParaformer

namespace = ""
# 加载环境变量

app = FastAPI()

if not os.path.exists(asr_model_path):
    logger.info("模型不存在，开始下载模型...")
    #模型下载
    from modelscope import snapshot_download
    model_dir = snapshot_download('iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',cache_dir="ckpt")

asr_model = SeacoParaformer(asr_model_path)

@app.post(namespace+"/asr",response_model=AsrResponseModel)
async def predict(request: AsrRequest):
    sessionId = request.sessionId
    audio_file_array_list = request.audio_array_list
    audio_file_array = [np.array(elem) for elem in audio_file_array_list]
    hotword = request.hotword
    start = time.time()
    try:
        result = asr_model(audio_file_array, hotword)
        logger.info(f"sessionId: {sessionId},ASR result: {result}")
        return AsrResponseModel(response=result, code=200, sessionId=sessionId,cost=time.time()-start)
    except Exception as error:
        logger.info(f"sessionId:{sessionId} Error processing ASR:{error}")
        return AsrResponseModel(response=error, code=500, sessionId=sessionId,cost=time.time()-start)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3012)

