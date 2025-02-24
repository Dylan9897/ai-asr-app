# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 15:32
from fastapi import FastAPI,Request
from funasr_onnx import CT_Transformer
from dotenv import load_dotenv
import uvicorn
import numpy as np
import traceback
import time

import os
import sys
sys.path.append(os.getcwd())

from utils.logger import logger
from utils.data_manager import PuncRequest,PuncResponseModel

namespace = ""
# 加载环境变量
load_dotenv()
app = FastAPI()
punc_model_path = os.getenv("punc_model_path")

if not os.path.exists(punc_model_path):
    logger.info("模型不存在，开始下载模型...")
    #模型下载
    from modelscope import snapshot_download
    model_dir = snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',cache_dir="ckpt")

punc_model = CT_Transformer(punc_model_path,quantize=False)
@app.post(namespace+"/punc",response_model=PuncResponseModel)
async def predict(request: PuncRequest):
    start = time.time()
    sessionId = request.sessionId
    raw_text = request.raw_text
    logger.info(f"sessionId:{sessionId},raw_text:{raw_text}")
    try:
        result = punc_model(raw_text)
        logger.info(f"sessionId:{sessionId},raw_text:{raw_text},result:{result}")
        return PuncResponseModel(sessionId=sessionId,code=200,response=result[0],cost=time.time()-start)
    except:
        error = traceback.format_exc()
        logger.info(f"sessionId:{sessionId},raw_text:{raw_text},error:{error}")
        return PuncResponseModel(sessionId=sessionId,code=500,response=error,cost=time.time()-start)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3013)
