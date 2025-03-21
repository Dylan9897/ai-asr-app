from fastapi import FastAPI,WebSocket
import uvicorn
import logging

from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        #manager.disconnect(websocket)
        print("客户端断开连接")
        logger.info(f"客户端断开连接: {websocket.client}")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8119)