# logger_config.py
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta


# 配置日志记录器
def setup_logger(log_file="runs/log/restapi.log"):
    handler = TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=15, encoding="UTF-8")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.WARNING)  # 设置处理程序的日志级别

    logger = logging.getLogger('object_detection')
    logger.setLevel(logging.WARNING)  # 设置记录器的日志级别
    logger.addHandler(handler)
    return logger


# 自定义记录日志的函数
def log_detection(logger, num_detection, alarm_status):
    current_time = datetime.now()
    global log_time
    log_interval = (current_time - log_time).total_seconds()
    if log_interval >= 60:  # 如果距离上次记录日志的时间超过60秒
        logger.warning(f'object detected, total {num_detection} objects')
        log_time = current_time  # 更新日志时间
        if alarm_status:
            logger.error('alarm activated')


# 初始化日志时间
log_time = datetime.now() - timedelta(minutes=1)
