import logging
import os
import sys
from datetime import datetime

def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOG_FILE_PATH = f'./log/app_{get_current_timestamp()}.txt'


def _setup_logger(name: str, level=logging.INFO, log_file=None) -> logging.Logger:
    '''
    Khởi tạo và trả về logger đã cấu hình:
    - Ghi ra console (hạn chế lỗi Unicode).
    - Ghi ra file log ./log/app_YYYY-MM-DD_HH-MM-SS.log với UTF-8.
    - Tự động tạo thư mục log nếu chưa tồn tại.
    '''
    formatter = logging.Formatter('<%(levelname)s-%(name)s> - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Không lặp log nếu dùng root logger

    if not logger.handlers:
        # Console handler: dùng stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File log path với timestamp

        # Tạo thư mục nếu chưa có
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # File handler UTF-8
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

if __name__ == "main":
    logger = _setup_logger("main", log_file="main.log")
    logger.debug("Debug message")
    logger.info("Info message")