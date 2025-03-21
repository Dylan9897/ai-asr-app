# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 13:59

from fastapi import FastAPI
from funasr_onnx import Fsmn_vad

import uvicorn
import os
import time
import sys
sys.path.append("/mnt/e/Github/ai-asr-app")
from utils.logger import logger
from utils.data_manager import VadRequest,VadResponseModel
from utils.config import vad_model_path

namespace = ""
# 加载环境变量
if not os.path.exists(vad_model_path):
    logger.info("模型不存在，开始下载模型...")
    #模型下载
    from modelscope import snapshot_download
    model_dir = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',cache_dir="ckpt")

vad_model = Fsmn_vad(vad_model_path,quantize=False)

app = FastAPI()
def process_vad_result(timestamps):
    # 展平嵌套列表
    flat_list = [item for sublist in timestamps for item in sublist]
    # 转换为元组
    result = [tuple(item) for item in flat_list]
    return result

@app.post(namespace+"/vad",response_model=VadResponseModel)
async def vad(request: VadRequest):
    start = time.time()
    sessionId = request.sessionId
    mono_channel_audio_path = request.file_path
    logger.info(f"sessionId: {sessionId}, mono_channel_audio_path: {mono_channel_audio_path}")

    if not mono_channel_audio_path:
        logger.info(f"sessionId: {sessionId},Error processing VAD: mono_channel_audio_path is empty")
        return VadResponseModel(sessionId=sessionId,code=500,response="mono_channel_audio_path is empty",cost=time.time()-start)
    try:
        vad_result = vad_model(mono_channel_audio_path)
        return VadResponseModel(sessionId=sessionId,code=200,response=vad_result,cost=time.time()-start)
    except Exception as error:
        logger.error(f"sessionId: {sessionId},Error processing VAD: {str(error)}")
        return VadResponseModel(sessionId=sessionId,code=500,response=error,cost=time.time()-start)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3011)



