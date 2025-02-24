#!/bin/bash

# 启动各个服务
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3011 components.vad.onnx_vad_api:app --timeout 100 &
#gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3012 components.asr.onnx_asr_api:app &
gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3013 components.punc.onnx_punc_api:app &
gunicorn -w 8 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:18022 main:app --timeout 1000 &
#python main.py &
# 等待所有后台进程退出（实际上不会到达这里，因为我们希望服务一直运行）
wait