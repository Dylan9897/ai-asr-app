# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/2/17 13:28
from loguru import logger

# 定义日志文件路径格式，使用{time}占位符来生成基于时间的文件名
log_file_path = "log/asr_{time:YYYY-MM-DD}.log"

# 添加sink，配置日志输出格式、轮换策略和保留策略
logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",  # 日志格式
    level="DEBUG",  # 设置最低日志级别
    rotation="00:00",  # 每天午夜进行日志轮换
    retention="30 days",  # 只保留最近30天的日志文件
    compression="zip"  # 轮换后压缩旧日志文件（可选）
)


