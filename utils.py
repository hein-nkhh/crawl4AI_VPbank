import json
from typing import List, Dict, Union
from logger import _setup_logger
import pandas as pd
import config
from dotenv import load_dotenv
import os
import time
from decimal import Decimal
from datetime import datetime, date

logger = _setup_logger(__name__, config.LOG_LEVEL)

class Utils:
    @staticmethod
    def load_json(file_path: str) -> dict:
        """Đọc file .json và trả về dữ liệu dưới dạng dict hoặc list."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Load FILE JSON {file_path} thành công")
                return data
        except Exception as e:
            logger.debug(f"Load FILE JSON {file_path} thất bại")
            raise Exception(e)

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Đọc file .jsonl và trả về list chứa các dict (mỗi dòng là một JSON).
    
        Args:
            file_path (str): Đường dẫn tới file .jsonl
            logger: Logger đã được cấu hình sẵn (thường dùng logging.getLogger(...))
    
        Returns:
            List[Dict]: Danh sách các dòng JSON đã parse thành dict
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.debug(f"Lỗi JSON ở dòng {line_num}: {e}")
        except FileNotFoundError:
            logger.debug(f"Không tìm thấy file: {file_path}")
        except Exception as e:
            logger.debug(f"Lỗi khi đọc file '{file_path}': {e}")
        return data

    @staticmethod
    def write_csv(data: List[Dict], file_path: str, encoding: str = "utf-8", index: bool = False) -> None:
        """
        Ghi list[dict] hoặc pandas.DataFrame ra file CSV.
    
        Args:
            data (List[Dict] or pd.DataFrame): Dữ liệu cần ghi.
            file_path (str): Đường dẫn file CSV đầu ra.
            encoding (str): Kiểu mã hóa (mặc định: "utf-8").
            index (bool): Có ghi chỉ số dòng (index) hay không (mặc định: False).
        """
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)
    
            df.to_csv(file_path, index=index, encoding=encoding)
            logger.debug(f"[write_csv] ✅ Đã ghi {len(df)} dòng vào '{file_path}'")
    
        except Exception as e:
            logger.debug(f"[write_csv] ❌ Lỗi khi ghi CSV: {e}")

    @staticmethod
    def load_api_key_from_env(key_name: str, env_path: str = ".env") -> str:
        """
        Load API key từ biến môi trường, tự động đọc từ file .env nếu chưa có.

        Args:
            key_name (str): Tên biến môi trường (ví dụ: "GEMINI_API_KEY").
            env_path (str): Đường dẫn tới file .env (mặc định là cùng thư mục).

        Returns:
            str: Giá trị API key

        Raises:
            EnvironmentError: Nếu không tìm thấy key trong môi trường
        """
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv(key_name)

        if not api_key:
            raise EnvironmentError(f"⚠️ Không tìm thấy biến môi trường: {key_name}")
        
        # logger.debug(f"[load_api_key_from_env] ✅ Đã load key '{key_name}' từ môi trường")
        return api_key
    
    @staticmethod
    def save_json(data: Union[Dict, List], file_path: str, ensure_ascii: bool = False, indent: int = 2) -> None:
        """
        Ghi dữ liệu (dict hoặc list) vào file JSON.

        Args:
            data (Union[Dict, List]): Dữ liệu cần ghi.
            file_path (str): Đường dẫn tới file JSON đầu ra.
            ensure_ascii (bool): Nếu True, các ký tự không ASCII sẽ được escape. Mặc định là False để giữ tiếng Việt.
            indent (int): Số khoảng trắng thụt dòng cho JSON đẹp.

        Raises:
            Exception: Nếu ghi file thất bại.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent, default=str)
            logger.debug(f"[save_json] ✅ Đã ghi JSON vào '{file_path}'")
        except Exception as e:
            logger.debug(f"[save_json] ❌ Lỗi khi ghi JSON: {e}")
            raise Exception(e)

    @staticmethod
    def get_s3_client(aws_access_key: str, aws_secret_key: str, region_name: str):
        """
        Tạo đối tượng boto3 S3 client từ thông tin cấu hình bắt buộc.
        """
        if not all([aws_access_key, aws_secret_key, region_name]):
            raise ValueError("⚠️ Cần truyền đủ aws_access_key, aws_secret_key và region_name.")

        return S3Connector(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        ).get_client()

    @staticmethod
    def fetch_json_from_s3(bucket: str, key: str, aws_access_key: str, aws_secret_key: str, region_name: str) -> Union[Dict, List]:
        """
        Tải và parse file JSON từ S3.
        """
        s3_client = Utils.get_s3_client(aws_access_key, aws_secret_key, region_name)
        fetcher = S3DataFetcher(s3_client)
        return fetcher.read_file(bucket_name=bucket, object_key=key, file_type="json")

    @staticmethod
    def fetch_csv_from_s3(bucket: str, key: str, aws_access_key: str, aws_secret_key: str, region_name: str) -> pd.DataFrame:
        """
        Tải file CSV từ S3 và trả về dưới dạng DataFrame.
        """
        s3_client = Utils.get_s3_client(aws_access_key, aws_secret_key, region_name)
        fetcher = S3DataFetcher(s3_client)
        return fetcher.read_file(bucket_name=bucket, object_key=key, file_type="csv")

    @staticmethod
    def fetch_text_from_s3(bucket: str, key: str, aws_access_key: str, aws_secret_key: str, region_name: str) -> str:
        """
        Tải file text từ S3 và trả về dưới dạng chuỗi.
        """
        s3_client = Utils.get_s3_client(aws_access_key, aws_secret_key, region_name)
        fetcher = S3DataFetcher(s3_client)
        return fetcher.read_file(bucket_name=bucket, object_key=key, file_type="text")


    @staticmethod
    def load_text(file_path: str, encoding: str = "utf-8") -> str:
        """
        Đọc file văn bản thuần (.txt) từ local và trả về nội dung dạng chuỗi.

        Args:
            file_path (str): Đường dẫn tới file .txt
            encoding (str): Mã hóa (mặc định: "utf-8")

        Returns:
            str: Nội dung văn bản

        Raises:
            Exception: Nếu đọc file thất bại
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                logger.debug(f"[load_text] ✅ Đã load file văn bản: {file_path}")
                return content
        except Exception as e:
            logger.debug(f"[load_text] ❌ Lỗi khi đọc file văn bản: {e}")
            raise Exception(e)

    @staticmethod
    def get_current_unix_time() -> str:
        """
        Lấy timestamp hiện tại theo định dạng YYYY-MM-DD_HH-MM-SS.
        """
        return int(time.time() * 1000000)
    
    @staticmethod
    def print_available_def(c) -> None:
        """
        In ra danh sách các method (function) có thể gọi được từ class hoặc instance.
        """
        for attr in dir(c):
            if callable(getattr(c, attr)) and not attr.startswith("__"):
                print(f"Method: {attr}")


    @staticmethod
    def json_to_str(data: dict | list, indent: int = 2) -> str:
        """
        Chuyển đổi object JSON (dict hoặc list) thành chuỗi JSON string.
        Tự động xử lý các kiểu không hỗ trợ như Decimal, datetime.
        """
        def default_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            return str(obj)

        try:
            return json.dumps(data, ensure_ascii=False, indent=indent, default=default_serializer)
        except Exception as e:
            return f"❌ Lỗi khi chuyển đổi JSON sang string: {e}"
        
    @staticmethod
    def json_str_to_dict(json_string: str):
        """
        Làm sạch chuỗi JSON (loại bỏ markdown ```json) và parse thành dict/list.
        """
        try:
            cleaned = json_string.strip()
            # Loại bỏ markdown nếu có
            if cleaned.startswith("```json"):
                cleaned = cleaned.removeprefix("```json").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned.removesuffix("```").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"❌ Lỗi khi parse JSON: {e}")
            return None