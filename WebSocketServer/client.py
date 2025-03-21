import asyncio
import websockets

WEBSOCKET_URL = "ws://192.168.1.101:8119/ws"

messages = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10"
]

async def send_messages_on_connection(websocket, messages):
    """在一个WebSocket连接上按顺序发送和接收消息"""
    for message in messages:
        await websocket.send(message)
        response = await websocket.recv()
        print(f"📩 发送: {message} | 📬 接收: {response}")

async def main():
    """并发发送多个 WebSocket 请求"""
    # 设置并发请求数量
    CONCURRENT_REQUESTS = 10

    # 分配消息到不同的连接
    messages_per_connection = [messages[i::CONCURRENT_REQUESTS] for i in range(CONCURRENT_REQUESTS)]

    # 创建WebSocket连接池
    websockets_pool = [websockets.connect(WEBSOCKET_URL) for _ in range(CONCURRENT_REQUESTS)]
    websockets_connections = await asyncio.gather(*websockets_pool)

    # 创建任务
    tasks = [
        send_messages_on_connection(websockets_connections[i], messages_per_connection[i])
        for i in range(CONCURRENT_REQUESTS)
    ]

    await asyncio.gather(*tasks)  # 并发执行所有任务

    # 关闭所有WebSocket连接
    for websocket in websockets_connections:
        await websocket.close()

asyncio.run(main())
