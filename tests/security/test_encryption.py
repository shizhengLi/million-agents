#!/usr/bin/env python3
"""
数据加密模块测试
测试AES加密、RSA加密、密钥管理等功能
"""

import pytest
import time
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.encryption import (
    DataEncryption, EncryptionError,
    AESCipher, RSACipher,
    KeyManager, KeyGenerationError
)


class TestKeyManager:
    """密钥管理器测试类"""

    @pytest.fixture
    def key_manager(self):
        """创建密钥管理器实例"""
        return KeyManager()

    def test_key_manager_initialization(self, key_manager):
        """测试密钥管理器初始化"""
        assert key_manager is not None
        assert hasattr(key_manager, '_master_key')
        assert hasattr(key_manager, '_aes_keys')
        assert hasattr(key_manager, '_rsa_keys')

    def test_generate_aes_key(self, key_manager):
        """测试生成AES密钥"""
        key = key_manager.generate_aes_key()

        assert isinstance(key, bytes)
        assert len(key) == 32  # AES-256
        assert key != key_manager.generate_aes_key()  # 每次生成的密钥应该不同

    def test_generate_aes_key_with_custom_length(self, key_manager):
        """测试生成自定义长度的AES密钥"""
        key_128 = key_manager.generate_aes_key(16)  # AES-128
        key_256 = key_manager.generate_aes_key(32)  # AES-256

        assert len(key_128) == 16
        assert len(key_256) == 32

    def test_generate_rsa_key_pair(self, key_manager):
        """测试生成RSA密钥对"""
        private_key, public_key = key_manager.generate_rsa_key_pair()

        assert private_key is not None
        assert public_key is not None
        assert private_key != public_key
        assert hasattr(private_key, 'private_bytes')
        assert hasattr(public_key, 'public_bytes')

    def test_generate_rsa_key_pair_with_custom_key_size(self, key_manager):
        """测试生成自定义密钥长度的RSA密钥对"""
        private_key_2048, public_key_2048 = key_manager.generate_rsa_key_pair(2048)
        private_key_4096, public_key_4096 = key_manager.generate_rsa_key_pair(4096)

        assert private_key_2048.key_size == 2048
        assert private_key_4096.key_size == 4096

    def test_store_and_retrieve_aes_key(self, key_manager):
        """测试存储和检索AES密钥"""
        key_id = "test_aes_key"
        key = key_manager.generate_aes_key()

        # 存储密钥
        key_manager.store_aes_key(key_id, key)

        # 检索密钥
        retrieved_key = key_manager.get_aes_key(key_id)
        assert retrieved_key == key

    def test_store_and_retrieve_rsa_key(self, key_manager):
        """测试存储和检索RSA密钥"""
        key_id = "test_rsa_key"
        private_key, public_key = key_manager.generate_rsa_key_pair()

        # 存储密钥
        key_manager.store_rsa_key_pair(key_id, private_key, public_key)

        # 检索密钥
        retrieved_private, retrieved_public = key_manager.get_rsa_key_pair(key_id)

        # 比较私钥
        from cryptography.hazmat.primitives import serialization
        orig_private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        retrieved_private_bytes = retrieved_private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        assert retrieved_private_bytes == orig_private_bytes

        # 比较公钥
        orig_public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        retrieved_public_bytes = retrieved_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        assert retrieved_public_bytes == orig_public_bytes

    def test_retrieve_nonexistent_key(self, key_manager):
        """测试检索不存在的密钥"""
        with pytest.raises(EncryptionError):
            key_manager.get_aes_key("nonexistent_key")

        with pytest.raises(EncryptionError):
            key_manager.get_rsa_key_pair("nonexistent_key")

    def test_key_rotation(self, key_manager):
        """测试密钥轮换"""
        key_id = "rotate_test_key"

        # 生成初始密钥
        initial_key = key_manager.generate_aes_key()
        key_manager.store_aes_key(key_id, initial_key)

        # 轮换密钥
        new_key = key_manager.rotate_aes_key(key_id)

        # 验证新密钥
        assert new_key != initial_key
        retrieved_key = key_manager.get_aes_key(key_id)
        assert retrieved_key == new_key

    def test_key_rotation_nonexistent_key(self, key_manager):
        """测试轮换不存在的密钥"""
        with pytest.raises(EncryptionError):
            key_manager.rotate_aes_key("nonexistent_key")

    def test_delete_key(self, key_manager):
        """测试删除密钥"""
        key_id = "delete_test_key"
        key = key_manager.generate_aes_key()
        key_manager.store_aes_key(key_id, key)

        # 删除密钥
        key_manager.delete_key(key_id)

        # 验证密钥已删除
        with pytest.raises(EncryptionError):
            key_manager.get_aes_key(key_id)

    def test_list_stored_keys(self, key_manager):
        """测试列出存储的密钥"""
        # 清空现有密钥
        key_manager._aes_keys.clear()
        key_manager._rsa_keys.clear()

        # 添加一些密钥
        key_manager.store_aes_key("aes_key1", b"key1")
        key_manager.store_aes_key("aes_key2", b"key2")
        private_key, public_key = key_manager.generate_rsa_key_pair()
        key_manager.store_rsa_key_pair("rsa_key1", private_key, public_key)

        # 列出密钥
        keys = key_manager.list_stored_keys()
        assert "aes_key1" in keys
        assert "aes_key2" in keys
        assert "rsa_key1" in keys


class TestAESCipher:
    """AES加密器测试类"""

    @pytest.fixture
    def aes_cipher(self):
        """创建AES加密器实例"""
        return AESCipher()

    def test_aes_encrypt_decrypt_string(self, aes_cipher):
        """测试AES加密解密字符串"""
        plaintext = "这是一个测试字符串"
        key = aes_cipher.generate_key()

        # 加密
        ciphertext = aes_cipher.encrypt(plaintext, key)

        # 解密
        decrypted = aes_cipher.decrypt(ciphertext, key)

        assert decrypted == plaintext

    def test_aes_encrypt_decrypt_bytes(self, aes_cipher):
        """测试AES加密解密字节"""
        plaintext = b"binary_data\x00\x01\x02"
        key = aes_cipher.generate_key()

        # 加密
        ciphertext = aes_cipher.encrypt(plaintext, key)

        # 解密
        decrypted = aes_cipher.decrypt(ciphertext, key)

        assert decrypted == plaintext

    def test_aes_encrypt_decrypt_with_different_keys(self, aes_cipher):
        """测试使用不同密钥的AES加密解密"""
        plaintext = "测试数据"
        key1 = aes_cipher.generate_key()
        key2 = aes_cipher.generate_key()

        # 使用key1加密
        ciphertext = aes_cipher.encrypt(plaintext, key1)

        # 使用key2解密应该失败
        with pytest.raises(EncryptionError):
            aes_cipher.decrypt(ciphertext, key2)

        # 使用key1解密应该成功
        decrypted = aes_cipher.decrypt(ciphertext, key1)
        assert decrypted == plaintext

    def test_aes_encrypt_empty_string(self, aes_cipher):
        """测试加密空字符串"""
        plaintext = ""
        key = aes_cipher.generate_key()

        ciphertext = aes_cipher.encrypt(plaintext, key)
        decrypted = aes_cipher.decrypt(ciphertext, key)

        assert decrypted == plaintext

    def test_aes_encrypt_large_data(self, aes_cipher):
        """测试加密大量数据"""
        plaintext = "A" * 10000  # 10KB数据
        key = aes_cipher.generate_key()

        ciphertext = aes_cipher.encrypt(plaintext, key)
        decrypted = aes_cipher.decrypt(ciphertext, key)

        assert decrypted == plaintext

    def test_aes_encrypt_with_invalid_key(self, aes_cipher):
        """测试使用无效密钥加密"""
        plaintext = "测试数据"
        invalid_keys = [
            None,
            "",
            b"short_key",
            "string_key",
            123
        ]

        for invalid_key in invalid_keys:
            with pytest.raises(EncryptionError):
                aes_cipher.encrypt(plaintext, invalid_key)

    def test_aes_decrypt_invalid_ciphertext(self, aes_cipher):
        """测试解密无效密文"""
        key = aes_cipher.generate_key()
        invalid_ciphertexts = [
            None,
            "",
            b"invalid",
            "not_base64",
            123
        ]

        for invalid_ciphertext in invalid_ciphertexts:
            with pytest.raises(EncryptionError):
                aes_cipher.decrypt(invalid_ciphertext, key)

    def test_aes_decrypt_with_wrong_key(self, aes_cipher):
        """测试使用错误密钥解密"""
        plaintext = "测试数据"
        key1 = aes_cipher.generate_key()
        key2 = aes_cipher.generate_key()

        ciphertext = aes_cipher.encrypt(plaintext, key1)

        with pytest.raises(EncryptionError):
            aes_cipher.decrypt(ciphertext, key2)

    def test_aes_key_generation(self, aes_cipher):
        """测试AES密钥生成"""
        key1 = aes_cipher.generate_key()
        key2 = aes_cipher.generate_key()

        assert isinstance(key1, bytes)
        assert len(key1) == 32
        assert key1 != key2

    def test_aes_performance(self, aes_cipher):
        """测试AES加密性能"""
        plaintext = "性能测试数据" * 1000  # 约16KB数据
        key = aes_cipher.generate_key()

        # 测试加密性能
        start_time = time.time()
        for _ in range(100):
            ciphertext = aes_cipher.encrypt(plaintext, key)
        encrypt_time = time.time() - start_time

        # 测试解密性能
        start_time = time.time()
        for _ in range(100):
            decrypted = aes_cipher.decrypt(ciphertext, key)
        decrypt_time = time.time() - start_time

        # 性能应该在合理范围内（1秒内完成100次加密/解密）
        assert encrypt_time < 1.0
        assert decrypt_time < 1.0


class TestRSACipher:
    """RSA加密器测试类"""

    @pytest.fixture
    def rsa_cipher(self):
        """创建RSA加密器实例"""
        return RSACipher()

    def test_rsa_generate_key_pair(self, rsa_cipher):
        """测试RSA密钥对生成"""
        private_key, public_key = rsa_cipher.generate_key_pair()

        assert private_key is not None
        assert public_key is not None
        assert private_key != public_key

    def test_rsa_encrypt_decrypt_small_data(self, rsa_cipher):
        """测试RSA加密解密小数据"""
        plaintext = "短数据"
        private_key, public_key = rsa_cipher.generate_key_pair()

        # 使用公钥加密
        ciphertext = rsa_cipher.encrypt(plaintext, public_key)

        # 使用私钥解密
        decrypted = rsa_cipher.decrypt(ciphertext, private_key)

        assert decrypted == plaintext

    def test_rsa_encrypt_decrypt_bytes(self, rsa_cipher):
        """测试RSA加密解密字节数据"""
        plaintext = b"binary_data"
        private_key, public_key = rsa_cipher.generate_key_pair()

        ciphertext = rsa_cipher.encrypt(plaintext, public_key)
        decrypted = rsa_cipher.decrypt(ciphertext, private_key)

        assert decrypted == plaintext

    def test_rsa_encrypt_large_data_should_fail(self, rsa_cipher):
        """测试RSA加密大数据应该失败"""
        # 生成大于RSA密钥长度的数据
        large_plaintext = "A" * 1000  # 1KB数据，对于2048位RSA来说太大了
        private_key, public_key = rsa_cipher.generate_key_pair()

        with pytest.raises(EncryptionError):
            rsa_cipher.encrypt(large_plaintext, public_key)

    def test_rsa_sign_verify(self, rsa_cipher):
        """测试RSA签名和验证"""
        message = "需要签名的消息"
        private_key, public_key = rsa_cipher.generate_key_pair()

        # 签名
        signature = rsa_cipher.sign(message, private_key)

        # 验证签名
        is_valid = rsa_cipher.verify(message, signature, public_key)

        assert is_valid is True

    def test_rsa_verify_invalid_signature(self, rsa_cipher):
        """测试验证无效签名"""
        message = "原始消息"
        wrong_message = "错误消息"
        private_key, public_key = rsa_cipher.generate_key_pair()

        # 对错误消息签名
        signature = rsa_cipher.sign(wrong_message, private_key)

        # 验证原始消息
        is_valid = rsa_cipher.verify(message, signature, public_key)

        assert is_valid is False

    def test_rsa_encrypt_with_wrong_key(self, rsa_cipher):
        """测试使用错误密钥加密"""
        plaintext = "测试数据"
        private_key1, public_key1 = rsa_cipher.generate_key_pair()
        private_key2, public_key2 = rsa_cipher.generate_key_pair()

        # 应该使用公钥加密，使用私钥解密
        with pytest.raises(EncryptionError):
            rsa_cipher.encrypt(plaintext, private_key1)  # 错误：使用私钥加密

    def test_rsa_decrypt_with_wrong_key(self, rsa_cipher):
        """测试使用错误密钥解密"""
        plaintext = "测试数据"
        private_key1, public_key1 = rsa_cipher.generate_key_pair()
        private_key2, public_key2 = rsa_cipher.generate_key_pair()

        ciphertext = rsa_cipher.encrypt(plaintext, public_key1)

        # 使用错误的私钥解密应该失败
        with pytest.raises(EncryptionError):
            rsa_cipher.decrypt(ciphertext, private_key2)

    def test_rsa_key_serialization(self, rsa_cipher):
        """测试RSA密钥序列化"""
        private_key, public_key = rsa_cipher.generate_key_pair()

        # 序列化密钥
        private_pem = rsa_cipher.serialize_private_key(private_key)
        public_pem = rsa_cipher.serialize_public_key(public_key)

        assert isinstance(private_pem, str)
        assert isinstance(public_pem, str)
        assert "BEGIN PRIVATE KEY" in private_pem
        assert "BEGIN PUBLIC KEY" in public_pem

    def test_rsa_key_deserialization(self, rsa_cipher):
        """测试RSA密钥反序列化"""
        private_key, public_key = rsa_cipher.generate_key_pair()

        # 序列化密钥
        private_pem = rsa_cipher.serialize_private_key(private_key)
        public_pem = rsa_cipher.serialize_public_key(public_key)

        # 反序列化密钥
        loaded_private = rsa_cipher.deserialize_private_key(private_pem)
        loaded_public = rsa_cipher.deserialize_public_key(public_pem)

        # 验证反序列化的密钥
        original_message = "测试消息"
        ciphertext = rsa_cipher.encrypt(original_message, loaded_public)
        decrypted = rsa_cipher.decrypt(ciphertext, loaded_private)

        assert decrypted == original_message

    def test_rsa_performance(self, rsa_cipher):
        """测试RSA加密性能"""
        plaintext = "性能测试数据"
        private_key, public_key = rsa_cipher.generate_key_pair()

        # 测试加密性能
        start_time = time.time()
        for _ in range(10):  # RSA比较慢，测试较少次数
            ciphertext = rsa_cipher.encrypt(plaintext, public_key)
        encrypt_time = time.time() - start_time

        # 测试解密性能
        start_time = time.time()
        for _ in range(10):
            decrypted = rsa_cipher.decrypt(ciphertext, private_key)
        decrypt_time = time.time() - start_time

        # RSA性能要求较宽松，5秒内完成10次操作
        assert encrypt_time < 5.0
        assert decrypt_time < 5.0


class TestDataEncryption:
    """数据加密主类测试"""

    @pytest.fixture
    def encryption(self):
        """创建数据加密实例"""
        return DataEncryption()

    def test_data_encryption_initialization(self, encryption):
        """测试数据加密类初始化"""
        assert encryption is not None
        assert encryption.key_manager is not None
        assert encryption.aes_cipher is not None
        assert encryption.rsa_cipher is not None

    def test_encrypt_sensitive_data(self, encryption):
        """测试加密敏感数据"""
        sensitive_data = "用户密码：123456"

        encrypted_data = encryption.encrypt_sensitive_data(sensitive_data)

        assert encrypted_data != sensitive_data
        assert isinstance(encrypted_data, str)
        assert "encrypted:" in encrypted_data

    def test_decrypt_sensitive_data(self, encryption):
        """测试解密敏感数据"""
        original_data = "用户密码：123456"
        encrypted_data = encryption.encrypt_sensitive_data(original_data)

        decrypted_data = encryption.decrypt_sensitive_data(encrypted_data)

        assert decrypted_data == original_data

    def test_encrypt_file_data(self, encryption):
        """测试加密文件数据"""
        file_content = b"file_content_data"
        file_path = "/path/to/secret/file.txt"

        encrypted_content, metadata = encryption.encrypt_file_data(file_content, file_path)

        assert encrypted_content != file_content
        assert isinstance(encrypted_content, bytes)
        assert "file_path" in metadata
        assert metadata["file_path"] == file_path

    def test_decrypt_file_data(self, encryption):
        """测试解密文件数据"""
        original_content = b"file_content_data"
        file_path = "/path/to/secret/file.txt"

        encrypted_content, metadata = encryption.encrypt_file_data(original_content, file_path)
        decrypted_content = encryption.decrypt_file_data(encrypted_content, metadata)

        assert decrypted_content == original_content

    def test_encrypt_database_field(self, encryption):
        """测试加密数据库字段"""
        field_value = "user_email@example.com"
        field_name = "email"
        table_name = "users"

        encrypted_value = encryption.encrypt_database_field(field_value, table_name, field_name)

        assert encrypted_value != field_value
        assert isinstance(encrypted_value, str)

    def test_decrypt_database_field(self, encryption):
        """测试解密数据库字段"""
        original_value = "user_email@example.com"
        table_name = "users"
        field_name = "email"

        encrypted_value = encryption.encrypt_database_field(original_value, table_name, field_name)
        decrypted_value = encryption.decrypt_database_field(encrypted_value, table_name, field_name)

        assert decrypted_value == original_value

    def test_hash_password(self, encryption):
        """测试密码哈希"""
        password = "user_password_123"

        hashed_password = encryption.hash_password(password)

        assert hashed_password != password
        assert isinstance(hashed_password, str)
        assert len(hashed_password) == 64  # SHA-256 hex length

    def test_verify_password(self, encryption):
        """测试密码验证"""
        password = "user_password_123"

        # 使用固定盐值生成哈希
        fixed_salt = "test_salt_123"
        hashed_password = encryption.hash_password(password, fixed_salt)

        # 正确密码验证
        is_valid = encryption.verify_password(password, hashed_password)
        assert is_valid is True

        # 错误密码验证
        is_invalid = encryption.verify_password("wrong_password", hashed_password)
        assert is_invalid is False

    def test_generate_secure_token(self, encryption):
        """测试生成安全令牌"""
        token1 = encryption.generate_secure_token()
        token2 = encryption.generate_secure_token()

        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert len(token1) == 32  # 默认长度
        assert token1 != token2

    def test_generate_secure_token_custom_length(self, encryption):
        """测试生成自定义长度的安全令牌"""
        token_16 = encryption.generate_secure_token(16)
        token_64 = encryption.generate_secure_token(64)

        assert len(token_16) == 16
        assert len(token_64) == 64

    def test_integrated_encryption_workflow(self, encryption):
        """测试完整的加密工作流"""
        # 模拟用户注册场景
        user_data = {
            "username": "testuser",
            "password": "secure_password_123",
            "email": "test@example.com",
            "phone": "1234567890"
        }

        # 1. 哈希密码（使用固定盐值用于测试）
        fixed_salt = "user_registration_salt"
        hashed_password = encryption.hash_password(user_data["password"], fixed_salt)

        # 2. 加密敏感字段
        encrypted_email = encryption.encrypt_database_field(
            user_data["email"], "users", "email"
        )
        encrypted_phone = encryption.encrypt_database_field(
            user_data["phone"], "users", "phone"
        )

        # 3. 验证数据可以正确解密
        decrypted_email = encryption.decrypt_database_field(
            encrypted_email, "users", "email"
        )
        decrypted_phone = encryption.decrypt_database_field(
            encrypted_phone, "users", "phone"
        )

        # 4. 验证密码
        is_password_valid = encryption.verify_password(
            user_data["password"], hashed_password
        )

        assert decrypted_email == user_data["email"]
        assert decrypted_phone == user_data["phone"]
        assert is_password_valid is True
        assert hashed_password != user_data["password"]

    def test_encryption_performance(self, encryption):
        """测试加密性能"""
        test_data = "性能测试数据" * 100

        # 测试批量加密性能
        start_time = time.time()
        encrypted_list = []
        for _ in range(100):
            encrypted = encryption.encrypt_sensitive_data(test_data)
            encrypted_list.append(encrypted)
        encrypt_time = time.time() - start_time

        # 测试批量解密性能
        start_time = time.time()
        for encrypted in encrypted_list:
            encryption.decrypt_sensitive_data(encrypted)
        decrypt_time = time.time() - start_time

        # 性能应该在合理范围内
        assert encrypt_time < 2.0
        assert decrypt_time < 2.0