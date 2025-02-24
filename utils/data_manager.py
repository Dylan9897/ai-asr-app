# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 13:34
from typing import List,Union

# 检验数据类型
from pydantic import BaseModel

class VadRequest(BaseModel):
    sessionId: str
    file_path: str

class VadResponseModel(BaseModel):
    sessionId: str
    code: int
    response: Union[str, List]
    cost:float

class AsrRequest(BaseModel):
    sessionId: str
    audio_array_list: List
    hotword:str

class AsrResponseModel(BaseModel):
    sessionId: str
    code: int
    response: Union[str, List]
    cost:float


class PuncRequest(BaseModel):
    sessionId: str
    raw_text: str

class PuncResponseModel(BaseModel):
    sessionId: str
    code: int
    response: Union[str, List]
    cost:float

class MainRequest(BaseModel):
    sessionId: str
    hotword:str
    audio_file_url: str

