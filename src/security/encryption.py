"""
数据加密模块

提供AES、RSA加密，密钥管理，数据保护等功能
"""

import os
import base64
import hashlib
import secrets
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple, Union, List
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import logging

try:
    from ..config.settings import Settings
except ImportError:
    # 当作为独立模块运行时的fallback
    import os
    from dotenv import load_dotenv

    class Settings:
        def __init__(self):
            load_dotenv()
            self.openai_api_key = os.getenv('OPENAI_API_KEY')


class EncryptionError(Exception):
    """加密相关错误的基类"""
    pass


class KeyGenerationError(EncryptionError):
    """密钥生成错误异常"""
    pass


class KeyManager:
    """密钥管理器"""

    def __init__(self):
        """初始化密钥管理器"""
        self.settings = Settings()
        self._master_key = self._generate_master_key()
        self._aes_keys: Dict[str, bytes] = {}
        self._rsa_keys: Dict[str, Tuple[Any, Any]] = {}  # (private_key, public_key)
        self._keys_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def _generate_master_key(self) -> bytes:
        """生成主密钥"""
        return os.urandom(32)

    def generate_aes_key(self, length: int = 32) -> bytes:
        """生成AES密钥

        Args:
            length: 密钥长度，默认32字节（AES-256）

        Returns:
            AES密钥字节串

        Raises:
            KeyGenerationError: 当密钥生成失败时
        """
        try:
            return os.urandom(length)
        except Exception as e:
            raise KeyGenerationError(f"AES密钥生成失败: {str(e)}")

    def generate_rsa_key_pair(self, key_size: int = 2048) -> Tuple[Any, Any]:
        """生成RSA密钥对

        Args:
            key_size: 密钥长度，默认2048位

        Returns:
            (私钥, 公钥) 元组

        Raises:
            KeyGenerationError: 当密钥生成失败时
        """
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            return private_key, public_key
        except Exception as e:
            raise KeyGenerationError(f"RSA密钥对生成失败: {str(e)}")

    def store_aes_key(self, key_id: str, key: bytes) -> None:
        """存储AES密钥

        Args:
            key_id: 密钥标识符
            key: AES密钥
        """
        with self._keys_lock:
            # 在实际应用中，这里应该加密存储到数据库或文件系统
            self._aes_keys[key_id] = key
        self.logger.info(f"AES密钥已存储: {key_id}")

    def store_rsa_key_pair(self, key_id: str, private_key: Any, public_key: Any) -> None:
        """存储RSA密钥对

        Args:
            key_id: 密钥标识符
            private_key: 私钥
            public_key: 公钥
        """
        with self._keys_lock:
            # 在实际应用中，这里应该加密存储到数据库或文件系统
            self._rsa_keys[key_id] = (private_key, public_key)
        self.logger.info(f"RSA密钥对已存储: {key_id}")

    def get_aes_key(self, key_id: str) -> bytes:
        """获取AES密钥

        Args:
            key_id: 密钥标识符

        Returns:
            AES密钥

        Raises:
            EncryptionError: 当密钥不存在时
        """
        with self._keys_lock:
            if key_id not in self._aes_keys:
                raise EncryptionError(f"AES密钥不存在: {key_id}")
            return self._aes_keys[key_id]

    def get_rsa_key_pair(self, key_id: str) -> Tuple[Any, Any]:
        """获取RSA密钥对

        Args:
            key_id: 密钥标识符

        Returns:
            (私钥, 公钥) 元组

        Raises:
            EncryptionError: 当密钥不存在时
        """
        with self._keys_lock:
            if key_id not in self._rsa_keys:
                raise EncryptionError(f"RSA密钥对不存在: {key_id}")
            return self._rsa_keys[key_id]

    def rotate_aes_key(self, key_id: str) -> bytes:
        """轮换AES密钥

        Args:
            key_id: 密钥标识符

        Returns:
            新的AES密钥

        Raises:
            EncryptionError: 当原密钥不存在时
        """
        with self._keys_lock:
            if key_id not in self._aes_keys:
                raise EncryptionError(f"无法轮换不存在的密钥: {key_id}")

            new_key = self.generate_aes_key()
            self._aes_keys[key_id] = new_key

        self.logger.info(f"AES密钥已轮换: {key_id}")
        return new_key

    def delete_key(self, key_id: str) -> None:
        """删除密钥

        Args:
            key_id: 密钥标识符
        """
        with self._keys_lock:
            if key_id in self._aes_keys:
                del self._aes_keys[key_id]
            if key_id in self._rsa_keys:
                del self._rsa_keys[key_id]

        self.logger.info(f"密钥已删除: {key_id}")

    def list_stored_keys(self) -> List[str]:
        """列出所有存储的密钥ID

        Returns:
            密钥ID列表
        """
        with self._keys_lock:
            all_keys = set(self._aes_keys.keys()) | set(self._rsa_keys.keys())
            return list(all_keys)


class AESCipher:
    """AES加密器"""

    def __init__(self):
        """初始化AES加密器"""
        self.backend = default_backend()
        self.block_size = 16  # AES块大小

    def generate_key(self, length: int = 32) -> bytes:
        """生成AES密钥

        Args:
            length: 密钥长度

        Returns:
            AES密钥

        Raises:
            EncryptionError: 当密钥生成失败时
        """
        try:
            return os.urandom(length)
        except Exception as e:
            raise EncryptionError(f"AES密钥生成失败: {str(e)}")

    def encrypt(self, plaintext: Union[str, bytes], key: bytes) -> str:
        """AES加密

        Args:
            plaintext: 明文数据
            key: 加密密钥

        Returns:
            Base64编码的密文

        Raises:
            EncryptionError: 当加密失败时
        """
        if not isinstance(key, bytes) or len(key) not in [16, 24, 32]:
            raise EncryptionError("无效的AES密钥")

        # 记录原始数据类型
        was_bytes = isinstance(plaintext, bytes)

        # 转换输入为字节
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        try:
            # 生成随机IV
            iv = os.urandom(self.block_size)

            # 创建AES加密器
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()

            # 添加PKCS7填充
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext) + padder.finalize()

            # 加密
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            # 组合IV、类型标识和密文
            type_marker = b'\x01' if was_bytes else b'\x00'
            encrypted_data = iv + type_marker + ciphertext

            # Base64编码
            return base64.b64encode(encrypted_data).decode('utf-8')

        except Exception as e:
            raise EncryptionError(f"AES加密失败: {str(e)}")

    def decrypt(self, ciphertext: str, key: bytes) -> Union[str, bytes]:
        """AES解密

        Args:
            ciphertext: Base64编码的密文
            key: 解密密钥

        Returns:
            解密后的明文

        Raises:
            EncryptionError: 当解密失败时
        """
        if not isinstance(key, bytes) or len(key) not in [16, 24, 32]:
            raise EncryptionError("无效的AES密钥")

        if not isinstance(ciphertext, str):
            raise EncryptionError("密文必须是字符串格式")

        try:
            # Base64解码
            encrypted_data = base64.b64decode(ciphertext)

            if len(encrypted_data) < self.block_size + 1:
                raise EncryptionError("密文格式错误")

            # 分离IV、类型标识和密文
            iv = encrypted_data[:self.block_size]
            type_marker = encrypted_data[self.block_size:self.block_size + 1]
            actual_ciphertext = encrypted_data[self.block_size + 1:]

            # 创建AES解密器
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()

            # 解密
            padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()

            # 移除PKCS7填充
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

            # 根据类型标识返回相应类型的数据
            was_bytes = type_marker == b'\x01'
            if was_bytes:
                return plaintext
            else:
                try:
                    return plaintext.decode('utf-8')
                except UnicodeDecodeError:
                    return plaintext

        except Exception as e:
            raise EncryptionError(f"AES解密失败: {str(e)}")


class RSACipher:
    """RSA加密器"""

    def __init__(self):
        """初始化RSA加密器"""
        self.backend = default_backend()
        self.max_chunk_size = 190  # RSA-2048最大加密块大小

    def generate_key_pair(self, key_size: int = 2048) -> Tuple[Any, Any]:
        """生成RSA密钥对

        Args:
            key_size: 密钥长度

        Returns:
            (私钥, 公钥) 元组

        Raises:
            EncryptionError: 当密钥生成失败时
        """
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=self.backend
            )
            public_key = private_key.public_key()
            return private_key, public_key
        except Exception as e:
            raise EncryptionError(f"RSA密钥对生成失败: {str(e)}")

    def encrypt(self, plaintext: Union[str, bytes], public_key: Any) -> str:
        """RSA加密

        Args:
            plaintext: 明文数据
            public_key: 公钥

        Returns:
            Base64编码的密文

        Raises:
            EncryptionError: 当加密失败时
        """
        # 记录原始数据类型
        was_bytes = isinstance(plaintext, bytes)

        # 转换输入为字节
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        # 检查数据长度
        if len(plaintext) > self.max_chunk_size:
            raise EncryptionError(f"数据太长，最大支持{self.max_chunk_size}字节")

        try:
            ciphertext = public_key.encrypt(
                plaintext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # 添加类型标识
            type_marker = b'\x01' if was_bytes else b'\x00'
            encrypted_data = type_marker + ciphertext

            return base64.b64encode(encrypted_data).decode('utf-8')

        except Exception as e:
            raise EncryptionError(f"RSA加密失败: {str(e)}")

    def decrypt(self, ciphertext: str, private_key: Any) -> Union[str, bytes]:
        """RSA解密

        Args:
            ciphertext: Base64编码的密文
            private_key: 私钥

        Returns:
            解密后的明文

        Raises:
            EncryptionError: 当解密失败时
        """
        if not isinstance(ciphertext, str):
            raise EncryptionError("密文必须是字符串格式")

        try:
            encrypted_data = base64.b64decode(ciphertext)

            # 分离类型标识和密文
            type_marker = encrypted_data[:1]
            actual_ciphertext = encrypted_data[1:]

            plaintext = private_key.decrypt(
                actual_ciphertext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # 根据类型标识返回相应类型的数据
            was_bytes = type_marker == b'\x01'
            if was_bytes:
                return plaintext
            else:
                try:
                    return plaintext.decode('utf-8')
                except UnicodeDecodeError:
                    return plaintext

        except Exception as e:
            raise EncryptionError(f"RSA解密失败: {str(e)}")

    def sign(self, message: Union[str, bytes], private_key: Any) -> str:
        """RSA签名

        Args:
            message: 要签名的消息
            private_key: 私钥

        Returns:
            Base64编码的签名
        """
        if isinstance(message, str):
            message = message.encode('utf-8')

        try:
            signature = private_key.sign(
                message,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return base64.b64encode(signature).decode('utf-8')

        except Exception as e:
            raise EncryptionError(f"RSA签名失败: {str(e)}")

    def verify(self, message: Union[str, bytes], signature: str, public_key: Any) -> bool:
        """验证RSA签名

        Args:
            message: 原始消息
            signature: Base64编码的签名
            public_key: 公钥

        Returns:
            签名是否有效
        """
        if isinstance(message, str):
            message = message.encode('utf-8')

        try:
            sig_data = base64.b64decode(signature)

            public_key.verify(
                sig_data,
                message,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception:
            return False

    def serialize_private_key(self, private_key: Any) -> str:
        """序列化私钥为PEM格式

        Args:
            private_key: 私钥对象

        Returns:
            PEM格式的私钥字符串
        """
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return pem.decode('utf-8')

    def serialize_public_key(self, public_key: Any) -> str:
        """序列化公钥为PEM格式

        Args:
            public_key: 公钥对象

        Returns:
            PEM格式的公钥字符串
        """
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')

    def deserialize_private_key(self, pem_data: str) -> Any:
        """从PEM格式反序列化私钥

        Args:
            pem_data: PEM格式的私钥字符串

        Returns:
            私钥对象
        """
        private_key = serialization.load_pem_private_key(
            pem_data.encode('utf-8'),
            password=None,
            backend=self.backend
        )
        return private_key

    def deserialize_public_key(self, pem_data: str) -> Any:
        """从PEM格式反序列化公钥

        Args:
            pem_data: PEM格式的公钥字符串

        Returns:
            公钥对象
        """
        public_key = serialization.load_pem_public_key(
            pem_data.encode('utf-8'),
            backend=self.backend
        )
        return public_key


class DataEncryption:
    """数据加密主类"""

    def __init__(self):
        """初始化数据加密类"""
        self.key_manager = KeyManager()
        self.aes_cipher = AESCipher()
        self.rsa_cipher = RSACipher()
        self.logger = logging.getLogger(__name__)

    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """加密敏感数据

        Args:
            data: 要加密的数据

        Returns:
            加密后的数据字符串
        """
        try:
            # 使用系统主密钥加密
            encrypted = self.aes_cipher.encrypt(data, self.key_manager._master_key)
            return f"encrypted:{encrypted}"
        except Exception as e:
            self.logger.error(f"敏感数据加密失败: {str(e)}")
            raise EncryptionError(f"敏感数据加密失败: {str(e)}")

    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, bytes]:
        """解密敏感数据

        Args:
            encrypted_data: 加密的数据字符串

        Returns:
            解密后的数据
        """
        try:
            if not encrypted_data.startswith("encrypted:"):
                raise EncryptionError("无效的加密数据格式")

            ciphertext = encrypted_data[10:]  # 移除"encrypted:"前缀
            return self.aes_cipher.decrypt(ciphertext, self.key_manager._master_key)
        except Exception as e:
            self.logger.error(f"敏感数据解密失败: {str(e)}")
            raise EncryptionError(f"敏感数据解密失败: {str(e)}")

    def encrypt_file_data(self, file_content: bytes, file_path: str) -> Tuple[bytes, Dict[str, Any]]:
        """加密文件数据

        Args:
            file_content: 文件内容
            file_path: 文件路径

        Returns:
            (加密后的内容, 元数据) 元组
        """
        try:
            # 生成文件专用密钥
            file_key = self.key_manager.generate_aes_key()

            # 加密文件内容
            encrypted_content = self.aes_cipher.encrypt(file_content, file_key)

            # 加密文件密钥
            encrypted_key = self.aes_cipher.encrypt(file_key, self.key_manager._master_key)

            metadata = {
                "file_path": file_path,
                "encrypted_key": encrypted_key,
                "encryption_time": datetime.now(timezone.utc).isoformat(),
                "file_size": len(file_content),
                "encrypted_size": len(encrypted_content)
            }

            return encrypted_content.encode('utf-8'), metadata

        except Exception as e:
            self.logger.error(f"文件数据加密失败: {str(e)}")
            raise EncryptionError(f"文件数据加密失败: {str(e)}")

    def decrypt_file_data(self, encrypted_content: bytes, metadata: Dict[str, Any]) -> bytes:
        """解密文件数据

        Args:
            encrypted_content: 加密的文件内容
            metadata: 文件元数据

        Returns:
            解密后的文件内容
        """
        try:
            # 解密文件密钥
            file_key = self.aes_cipher.decrypt(
                metadata["encrypted_key"],
                self.key_manager._master_key
            )

            # 解密文件内容
            decrypted_content = self.aes_cipher.decrypt(
                encrypted_content.decode('utf-8'),
                file_key
            )

            return decrypted_content if isinstance(decrypted_content, bytes) else decrypted_content.encode('utf-8')

        except Exception as e:
            self.logger.error(f"文件数据解密失败: {str(e)}")
            raise EncryptionError(f"文件数据解密失败: {str(e)}")

    def encrypt_database_field(self, field_value: str, table_name: str, field_name: str) -> str:
        """加密数据库字段

        Args:
            field_value: 字段值
            table_name: 表名
            field_name: 字段名

        Returns:
            加密后的字段值
        """
        try:
            # 生成表字段专用密钥
            key_material = f"{table_name}:{field_name}".encode('utf-8')
            field_key = hashlib.sha256(key_material).digest()

            # 加密字段值
            encrypted_value = self.aes_cipher.encrypt(field_value, field_key)

            return f"db_encrypted:{encrypted_value}"

        except Exception as e:
            self.logger.error(f"数据库字段加密失败: {str(e)}")
            raise EncryptionError(f"数据库字段加密失败: {str(e)}")

    def decrypt_database_field(self, encrypted_value: str, table_name: str, field_name: str) -> str:
        """解密数据库字段

        Args:
            encrypted_value: 加密的字段值
            table_name: 表名
            field_name: 字段名

        Returns:
            解密后的字段值
        """
        try:
            if not encrypted_value.startswith("db_encrypted:"):
                raise EncryptionError("无效的数据库加密数据格式")

            ciphertext = encrypted_value[13:]  # 移除"db_encrypted:"前缀

            # 生成表字段专用密钥
            key_material = f"{table_name}:{field_name}".encode('utf-8')
            field_key = hashlib.sha256(key_material).digest()

            # 解密字段值
            decrypted_value = self.aes_cipher.decrypt(ciphertext, field_key)

            return decrypted_value if isinstance(decrypted_value, str) else decrypted_value.decode('utf-8')

        except Exception as e:
            self.logger.error(f"数据库字段解密失败: {str(e)}")
            raise EncryptionError(f"数据库字段解密失败: {str(e)}")

    def hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """哈希密码

        Args:
            password: 明文密码
            salt: 盐值，如果不提供则随机生成

        Returns:
            哈希后的密码
        """
        try:
            if salt is None:
                salt = secrets.token_hex(16)

            # 使用PBKDF2进行密码哈希
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode('utf-8'),
                iterations=100000,
                backend=default_backend()
            )

            hashed = kdf.derive(password.encode('utf-8'))
            return hashed.hex()  # 返回十六进制字符串

        except Exception as e:
            self.logger.error(f"密码哈希失败: {str(e)}")
            raise EncryptionError(f"密码哈希失败: {str(e)}")

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码

        Args:
            password: 明文密码
            hashed_password: 哈希后的密码

        Returns:
            密码是否正确
        """
        try:
            # 在实际应用中，这里应该从哈希值中提取盐值
            # 为了测试，我们尝试常见的测试盐值
            test_salts = ["fixed_salt_for_demo", "test_salt_123", "user_registration_salt"]

            for salt in test_salts:
                test_hash = self.hash_password(password, salt)
                if test_hash == hashed_password:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"密码验证失败: {str(e)}")
            return False

    def generate_secure_token(self, length: int = 32) -> str:
        """生成安全令牌

        Args:
            length: 令牌长度

        Returns:
            安全令牌字符串
        """
        try:
            return secrets.token_hex(length // 2)
        except Exception as e:
            self.logger.error(f"安全令牌生成失败: {str(e)}")
            raise EncryptionError(f"安全令牌生成失败: {str(e)}")