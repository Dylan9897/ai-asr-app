# Paraformer ASR Service with ONNX

​	本项目旨在提供一个高效、稳定的自动语音识别（ASR）服务，使用了Paraformer系列模型，并特别针对CPU环境优化了推理速度。该服务支持多线程部署，并通过分离音频声道来分别处理VAD（Voice Activity Detection）、ASR和标点预测任务。

## 特性

- **ONNX模型**：在CPU上的推理速度达到1:8的加速比。
- **多线程部署**：利用Gunicorn实现高效的多线程服务部署。
- **Paraformer模型**：选用了Paraformer系列模型，专注于ASR、VAD及标点预测功能，舍弃了人声判别模块。
- **自动化模型下载**：所需模型可自动下载并更新。
- **双声道音频处理**：输入为双声道音频，服务会先将音频切分为单声道，然后分别进行处理。
- **缓存管理**：处理完成后自动删除本地缓存的音频文件以节省空间。

## 安装与启动

### 安装步骤

1. 克隆仓库到本地：
   
   ```bash
   git clone ai-asr-app
   cd ai-asr-app
   pip install -r requirements.txt
   ```
   
1. 启动服务
   
   ```bash
   #!/bin/bash
   
   # 启动各个服务
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3011 components.vad.onnx_vad_api:app --timeout 100 &
   gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3013 components.punc.onnx_punc_api:app &
   gunicorn -w 8 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:18022 main:app --timeout 1000 &
   
   # 等待所有后台进程退出（实际上不会到达这里，因为我们希望服务一直运行）
   wait

## 更新日志 (Changelog)

### [v0.0.1] - 2025-03-03
- **音频格式转换**：首先将压缩的MP3文件转换为WAV格式。这是因为WAV文件未经压缩，更适合用于精确的语音活动检测（VAD）和后续的推理过程。
- **时间戳读取与分割**：根据VAD模型预测的时间戳来读取并分割单个音频数据，将其转换为nd.array数组（原生的SeacoParaformer模型智能batch推理音频文件）。
- **批量推理**：使用转换后的音频数据进行批量推理，从而提高处理效率。
