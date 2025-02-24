import requests
url = "http://0.0.0.0:18022/predict"

mp3_audio_file_url = "http://221.6.195.22:8092/recordings/2025-02-06/2025-02-06-15-26-43-7589_8230_018052676818.mp3"

long_audio_file_url = 'http://221.6.195.22:19080/recordings/2025-02-06/2025-02-06-15-21-26-6628_5502_015627699350.wav'
headers = {
    "Content-Type": "application/json",
}

data = {
    "sessionId":"123",
    "audio_file_url": long_audio_file_url,
    "hotword": "",
}
for i in range(10):
    response = requests.post(url=url, json=data, headers=headers)
    print(response.json())