import aiohttp
import asyncio
import pandas as pd
import time

audio_address_file_path = 'audio_address.xlsx'
url = "http://192.168.1.101:18003/predict"

headers = {
    "Content-Type": "application/json",
}

# 设置并发请求数量
CONCURRENT_REQUESTS = 10

async def fetch(session, audio_url, semaphore):
    start_time = time.time()
    async with semaphore:
        data = {
            "sessionId": "123",
            "audio_file_url": audio_url,
            "hotword": "",
        }
        try:
            async with session.post(url, json=data, headers=headers) as response:
                response_time = time.time() - start_time
                if response.status == 200:
                    response_json = await response.json()
                    return True, response_json, response_time
                else:
                    response_json = await response.json()
                    print(f"Failed: {data}")
                    return False, response_json, response_time
        except Exception as e:
            print(f"Failed: {data}")
            response_time = time.time() - start_time
            return False, str(e), response_time

async def main():
    data_frame = pd.read_excel(audio_address_file_path, sheet_name='Sheet2')
    audio_file_url_list = data_frame.iloc[:, 0].to_list()
    audio_file_url_list = [
        'http://106.15.137.87:81/recordings/2025-02-11/2025-02-11-17-14-50-6735_89727869.mp3',
        'http://106.15.137.87:81/recordings/2025-02-18/2025-02-18-13-43-22-7536_89727869.mp3',
        'http://106.15.137.87:81/recordings/2025-01-15/2025-01-15-11-00-59-2573_89727871.mp3',
        'http://106.15.137.87:81/recordings/2025-02-07/2025-02-07-11-03-40-4307_89727871.mp3',
        'http://106.15.137.87:81/recordings/2025-02-12/2025-02-12-15-08-46-2556_89727871.mp3',
        'http://106.15.137.87:81/recordings/2025-02-10/2025-02-10-15-45-15-9292_89727871.mp3',
        'http://106.15.137.87:81/recordings/2025-02-24/2025-02-24-10-50-08-1876_89727874.mp3',
        'http://106.15.137.87:81/recordings/2025-01-03/2025-01-03-15-59-36-2711_89727884.mp3'
    ]
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, audio_url, semaphore) for audio_url in audio_file_url_list]
        results = await asyncio.gather(*tasks)

    success_count = 0
    failure_count = 0
    total_response_time = 0.0

    for success, response, response_time in results:
        total_response_time += response_time
        if success:
            success_count += 1
            print(f"Success: {response}")
        else:
            failure_count += 1
            print(f"Failure: {response}")

    print(f"Total Requests: {len(results)}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {failure_count}")


if __name__ == "__main__":
    asyncio.run(main())

