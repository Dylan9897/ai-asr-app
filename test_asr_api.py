import requests
import pandas as pd

audio_address_file_path = 'audio_address.xlsx'
url = "http://192.168.1.101:18011/predict"


headers = {
    "Content-Type": "application/json",
}


data_frame = pd.read_excel(audio_address_file_path,sheet_name='Sheet2')

audio_file_url_list = data_frame.iloc[:,0].to_list()

for audio_url in audio_file_url_list:
    data = {
        "sessionId":"123",
        "audio_file_url": audio_url,
        "hotword": "",
}
    response = requests.post(url=url, json=data, headers=headers)
    print(response.json())