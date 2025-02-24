# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 13:55

import requests


def request_vad(input_data):
    url = "http://0.0.0.0:3021/vad"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=url,json=input_data,headers=headers)
    response_data = response.json()
    return response_data

def request_asr(input_data):
    url = "http://0.0.0.0:302/asr"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=url,json=input_data,headers=headers)
    response_data = response.json()
    return response_data

def request_punc(input_data):
    url = "http://0.0.0.0:3023/punc"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url=url,json=input_data,headers=headers)
    response_data = response.json()
    return response_data
