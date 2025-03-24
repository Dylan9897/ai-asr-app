# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/3/24 15:41
import sys
import json
import wget
import numpy as np
import traceback
import os
import soundfile as sf
from quart import Quart, request
from funasr import AutoModel
from utils.logger import logger

app = Quart(__name__)

# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

model_1 = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                    spk_model="cam++", spk_model_revision="v2.0.2",
                    )


def numpy_json_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # 处理数组
        return obj.tolist()
    return json.JSONEncoder().default(obj)


def split_stereo_to_mono(input_file, output_left, output_right):
    # 读取音频文件
    data, samplerate = sf.read(input_file)
    # 检查是否为双声道
    if data.shape[1] != 2:
        print("single channel")
        return False
    # 分离声道
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    # 导出单声道音频文件
    sf.write(output_left, left_channel, samplerate)
    sf.write(output_right, right_channel, samplerate)
    return True


def judge_isnull(param):
    if param == []:
        return True

    if len(param) >= 0 and "sentence_info" not in param[0]:
        return True
    return False


def get_ocr_result(file_url, hotword, sessionId):
    mapping = {
        "0": "0",
        "1": "1"
    }
    #
    filename = wget.download(file_url)

    name = filename.split(".")[0]

    output_left = f"{name}_left.wav"
    output_right = f"{name}_right.wav"

    try:
        status = split_stereo_to_mono(filename, output_left, output_right)
    except:
        status = False

    if status == False:
        logger.info("音频文件不是双声道")
        result = model_1.generate(input=filename,
                                  batch_size_s=1,
                                  hotword=hotword)

        logger.info(f"sessionId: {sessionId},result: {result}")
        try:
            if len(result) > 0:
                os.remove(filename)
                return result[0]["sentence_info"]
            else:
                return []
        except:
            return []

    else:
        left_result = model_1.generate(input=output_left,
                                       batch_size_s=1,
                                       hotword=hotword)

        right_result = model_1.generate(input=output_right,
                                        batch_size_s=1,
                                        hotword=hotword)

        os.remove(filename)
        # 删除中间文件
        os.remove(output_left)
        os.remove(output_right)

        logger.info(f"sessionId:{sessionId},left_result:{left_result}, right_result:{right_result}")
        # 如果右声道没声音，则返回左声道结果
        if judge_isnull(right_result) and not judge_isnull(left_result):
            left_result_info = left_result[0]["sentence_info"]
            for i, unit in enumerate(left_result_info):
                left_result_info[i]["spk"] = "0"
            return left_result_info
        # 如果左声道没声音，则返回右声道结果
        if judge_isnull(left_result) and not judge_isnull(right_result):
            right_result_info = right_result[0]["sentence_info"]
            for i, unit in enumerate(right_result_info):
                right_result_info[i]["spk"] = "0"
            return right_result_info

        # 如果左右声道都没有有声音，则返回空
        if judge_isnull(left_result) and judge_isnull(right_result):
            return []

        left_result_info = left_result[0]["sentence_info"]
        right_result_info = right_result[0]["sentence_info"]
        for i, unit in enumerate(left_result_info):
            left_result_info[i]["spk"] = "0"

        for i, unit in enumerate(right_result_info):
            right_result_info[i]["spk"] = "1"

        result_info = left_result_info + right_result_info

        result_info = sorted(result_info, key=lambda x: x["start"])
        return result_info


@app.route('/predict', methods=['POST'])
async def predict():
    data = await request.get_json()
    sessionId = data["sessionId"]
    hotword = data["hotword"]
    rd = data['audio_file']
    logger.info(f"request data: {data}")
    try:

        result = get_ocr_result(rd, hotword, sessionId)
        return str({"result": result, "code": 200, "sessionId": sessionId})

    except:
        error = traceback.format_exc()
        logger.info(f"sessionId: {sessionId},error: {error}")
        return str({"result": [], "code": 500, "sessionId": sessionId})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=sys.argv[1])