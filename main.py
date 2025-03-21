import os
import time
import traceback

from fastapi import FastAPI
import asyncio
import uvicorn
import aiohttp
from utils.logger import logger
from utils.config import *
from utils.data_manager import MainRequest
from utils.utils import download_audio_from_url,split_stereo_to_mono,merge_and_sort_timestamps,audio_segments,convert_mp3_to_wav
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
async def predict(request:MainRequest):
    sessionId = request.sessionId
    audio_file_url = request.audio_file_url
    hotword = request.hotword
    logger.info(f"sessionId:{sessionId},audio_file_url:{audio_file_url},hotword:{hotword}")
    start_time = time.time()
    try:
        logger.info(f"sessionId is {sessionId}, start downloading file:{audio_file_url}")
        # 下载录音到本地
        asr_final_result = []
        async with aiohttp.ClientSession() as session:
            tasks = download_audio_from_url(session, audio_file_url, raw_audio_dir)
            result = await asyncio.gather(tasks)
            logger.info(f"sessionId:{sessionId}, 下载音频文件完成，耗时：{time.time() - start_time}")
        download_finish_time = time.time()
        if result[0][1]:
            audio_file_path = result[0][0]
            if audio_file_path.endswith(".mp3"):
                convert_result = convert_mp3_to_wav(audio_file_path, audio_file_path.replace(".mp3", ".wav"))
                if convert_result:
                    os.remove(audio_file_path)
                    audio_file_path = audio_file_path.replace(".mp3", ".wav")

            audio_file_name = os.path.basename(audio_file_path)

            logger.info(f"sessionId:{sessionId}, 开始切分音频文件：{audio_file_name}")
            file_name, file_extension = os.path.splitext(audio_file_name)
            left_audio_file_name = f"{file_name}_left{file_extension}"
            right_audio_file_name = f"{file_name}_right{file_extension}"
            segment_result = split_stereo_to_mono(audio_file_path, left_audio_file_name, right_audio_file_name)

            logger.info(f"sessionId:{sessionId}, 左声道文件名：{left_audio_file_name}")
            logger.info(f"sessionId:{sessionId}, 右声道文件名：{right_audio_file_name}")
            if segment_result:
                left_audio_segment_path = os.path.join(segment_audio_dir, left_audio_file_name)
                right_audio_segment_path = os.path.join(segment_audio_dir, right_audio_file_name)
                vad_start_time = time.time()

                logger.info(f"sessionId:{sessionId}, 左声道文件地址：{left_audio_segment_path}")
                left_audio_time_stamp = request_vad({"sessionId": sessionId, "file_path": left_audio_segment_path})
                if left_audio_time_stamp["code"] != 200:
                    left_audio_time_stamp = []
                else:
                    left_audio_time_stamp = left_audio_time_stamp["response"][0]

                logger.info(f"sessionId:{sessionId}, 右声道文件地址：{right_audio_segment_path}")
                right_audio_time_stamp = request_vad(
                    {"sessionId": sessionId, "file_path": right_audio_segment_path})
                if right_audio_time_stamp["code"] != 200:
                    right_audio_time_stamp = []
                else:
                    right_audio_time_stamp = right_audio_time_stamp["response"][0]
                logger.info(f"vad time is {time.time() - vad_start_time}")
                merged_time_stamp = merge_and_sort_timestamps(left_audio_time_stamp, right_audio_time_stamp)
                segments = audio_segments(left_audio_segment_path, right_audio_segment_path, merged_time_stamp)
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
            else:
                logger.error({"sessionId": sessionId, "response": "切分音频文件失败", "code": 500, "cost": time.time() - start_time})
                return {"sessionId": sessionId, "response": "切分音频文件失败", "code": 500, "cost": time.time() - start_time}

            return {"sessionId": sessionId, "response": asr_final_result, "code": 200, "all cost time": time.time() - start_time,"download audio time":download_finish_time - start_time,"asr time":asr_time}

        else:
            logger.error({"sessionId": sessionId, "response": "下载音频文件失败", "code": 500, "cost": time.time() - start_time})
            return {"sessionId": sessionId, "response": "下载音频文件失败", "code": 500, "cost": time.time() - start_time}

    except:
        error = traceback.format_exc()
        logger.info(f"Error, sessionId is {sessionId}, error is {error}")
        return {"sessionId": sessionId, "response": error, "code": 500, "cost": time.time() - start_time}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=18003)

