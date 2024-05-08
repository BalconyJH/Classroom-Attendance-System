from pathlib import Path

from loguru import logger
from cryptography.fernet import Fernet


class DataProcess:
    def __init__(self, key_path: str = "key.key"):
        self.key_path = Path(key_path)
        self.key = self.load_or_generate_key()
        self.cipher = Fernet(self.key)

    def load_or_generate_key(self) -> bytes:
        logger.debug(f"Loading key from {self.key_path.absolute()}")
        try:
            with open(self.key_path, "rb") as file:
                return file.read()
        except FileNotFoundError:
            logger.debug(f"Key file not found at {self.key_path.absolute()}, generating a new key")
            new_key = Fernet.generate_key()
            with open(self.key_path, "wb") as file:
                file.write(new_key)
            return new_key

    def encrypt(self, data: str) -> bytes:
        """
        加密给定的字符串数据。

        :param data: 要加密的字符串。
        :return: 加密后的数据（字节字符串）。
        """
        return self.cipher.encrypt(data.encode())

    def decrypt(self, token: bytes) -> str:
        """
        解密给定的加密数据。

        :param token: 要解密的字节字符串。
        :return: 解密后的原始字符串。
        """
        return self.cipher.decrypt(token).decode()
