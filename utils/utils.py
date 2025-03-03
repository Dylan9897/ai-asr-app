# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 13:39
import os
import asyncio
import aiohttp
import traceback
import librosa
import soundfile as sf
import numpy as np

from urllib.parse import urlparse, unquote

from .logger import logger
from .config import segment_audio_dir

async def download_audio_from_url(session,audio_url,raw_audio_dir,max_retries=3):
    # 确保临时保存的目录存在
    if not os.path.exists(raw_audio_dir):
        os.makedirs(raw_audio_dir)
    attempt = 0
    while attempt < max_retries:
        try:
            async with session.get(audio_url,timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    parsed_url = urlparse(audio_url)
                    filename = os.path.basename(unquote(parsed_url.path))
                    # 将音频文件保存到本地
                    with open(f"{raw_audio_dir}/{filename}", "wb") as f:
                        f.write(await response.read())
                        return f"{raw_audio_dir}/{filename}",True
        except asyncio.TimeoutError:
            attempt += 1
        except aiohttp.ClientError as e:
            logger.error(f"Error downloading audio from {audio_url}: {e}")
            return None, False
    logger.error(f"Error downloading audio from {audio_url}")
    return None, False


def convert_mp3_to_wav(mp3_file_path, wav_file_path, samplerate=16000):
    """
    将mp3文件转换为wav文件

    :param mp3_file_path: 输入的mp3文件路径
    :param wav_file_path: 输出的wav文件路径
    :param samplerate: 采样率，默认为16000
    :return: 转换成功返回True，失败返回False
    """
    try:
        # 读取mp3文件
        data, _ = sf.read(mp3_file_path)
        # 保存为wav文件
        sf.write(wav_file_path, data, samplerate)
        return True
    except Exception as e:
        logger.error(f"Error converting mp3 to wav: {e}")
        return False


def split_stereo_to_mono(input_file, left_audio_file_name, right_audio_file_name):
    left_audio_save_path = os.path.join(segment_audio_dir, left_audio_file_name)
    right_audio_save_path = os.path.join(segment_audio_dir, right_audio_file_name)
    try:
        data, samplerate = sf.read(input_file)
        if data.shape[1] != 2:
            return False
        left_channel = data[:, 0]
        right_channel = data[:, 1]

        sf.write(left_audio_save_path, left_channel, samplerate)
        sf.write(right_audio_save_path, right_channel, samplerate)
        return True
    except:
        error = traceback.format_exc()
        logger.info(f"Error is exception, error is {error}")
        return False
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)


def merge_and_sort_timestamps(left_time_tuple,right_time_tuple):
    # 按照时间戳合并音频文件
    # 合并时间戳并标记来源
    merged_timestamps = []
    for start, end in left_time_tuple:
        merged_timestamps.append((start, end, "left"))
    for start, end in right_time_tuple:
        merged_timestamps.append((start, end, "right"))

    # 按开始时间排序
    merged_timestamps.sort(key=lambda x: x[0])
    return merged_timestamps

def load_wav(path: str, start_ms: int, end_ms: int) -> np.ndarray:
    # 将时间戳从毫秒转换为秒
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    duration_sec = end_sec - start_sec

    # 读取音频片段
    waveform, sr = librosa.load(path, sr=16000, offset=start_sec, duration=duration_sec)
    return waveform

def audio_segments(left_wav_path,right_wav_path,merged_time_stamp):
    """
    按时间戳读取音频文件
    :param left_wav_path:
    :param right_wav_path:
    :param merged_time_stamp:
    :return:
    """
    segments = []
    speaker_mapping = {
        "left":None,
        "right":None
    }
    for index,item in enumerate(merged_time_stamp):
        start, end, spker = item
        if index == 0:
            if spker == "left":
                speaker_mapping["left"] = "speaker_0"
                speaker_mapping["right"] = "speaker_1"
            else:
                speaker_mapping["right"] = "speaker_0"
                speaker_mapping["left"] = "speaker_1"
        if spker == "left":
            segment = load_wav(left_wav_path, start, end)
        if spker == "right":
            segment = load_wav(right_wav_path, start, end)
        segments.append({"spk":speaker_mapping[spker], "audio": segment, "start": start, "end": end})
    if os.path.exists(left_wav_path):
        os.remove(left_wav_path)
    if os.path.exists(right_wav_path):
        os.remove(right_wav_path)
    return segments

if __name__ == '__main__':
    audio_file_path = ""