#!/usr/bin/env python3
"""
JWT认证系统测试
测试JWT token的生成、验证、刷新等功能
"""

import pytest
import time
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.jwt_auth import JWTAuthenticator, TokenExpired, InvalidToken, AuthenticationError


class TestJWTAuthenticator:
    """JWT认证器测试类"""

    @pytest.fixture
    def jwt_auth(self):
        """创建JWT认证器实例"""
        return JWTAuthenticator(
            secret_key="test_secret_key_for_testing_only",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )

    def test_jwt_authenticator_initialization(self, jwt_auth):
        """测试JWT认证器初始化"""
        assert jwt_auth.secret_key == "test_secret_key_for_testing_only"
        assert jwt_auth.algorithm == "HS256"
        assert jwt_auth.access_token_expire_minutes == 30
        assert jwt_auth.refresh_token_expire_days == 7

    def test_jwt_authenticator_initialization_with_default_values(self):
        """测试JWT认证器使用默认值初始化"""
        auth = JWTAuthenticator()
        assert auth.secret_key is not None
        assert auth.algorithm == "HS256"
        assert auth.access_token_expire_minutes == 30
        assert auth.refresh_token_expire_days == 7

    def test_generate_access_token(self, jwt_auth):
        """测试生成访问令牌"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        token = jwt_auth.generate_access_token(user_data)

        assert isinstance(token, str)
        assert len(token) > 0
        assert "." in token  # JWT token格式: header.payload.signature

    def test_generate_refresh_token(self, jwt_auth):
        """测试生成刷新令牌"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser"
        }

        token = jwt_auth.generate_refresh_token(user_data)

        assert isinstance(token, str)
        assert len(token) > 0
        assert "." in token

    def test_generate_token_pair(self, jwt_auth):
        """测试生成令牌对"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        tokens = jwt_auth.generate_token_pair(user_data)

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "token_type" in tokens
        assert "expires_in" in tokens

        assert tokens["token_type"] == "bearer"
        assert isinstance(tokens["access_token"], str)
        assert isinstance(tokens["refresh_token"], str)
        assert tokens["expires_in"] == jwt_auth.access_token_expire_minutes * 60

    def test_verify_valid_access_token(self, jwt_auth):
        """测试验证有效的访问令牌"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "role": "user"
        }

        token = jwt_auth.generate_access_token(user_data)
        payload = jwt_auth.verify_token(token)

        assert payload["user_id"] == "test_user_123"
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert "exp" in payload
        assert "iat" in payload
        assert "type" in payload
        assert payload["type"] == "access"

    def test_verify_valid_refresh_token(self, jwt_auth):
        """测试验证有效的刷新令牌"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser"
        }

        token = jwt_auth.generate_refresh_token(user_data)
        payload = jwt_auth.verify_token(token)

        assert payload["user_id"] == "test_user_123"
        assert payload["username"] == "testuser"
        assert "type" in payload
        assert payload["type"] == "refresh"

    def test_verify_invalid_token(self, jwt_auth):
        """测试验证无效令牌"""
        invalid_tokens = [
            "",
            "invalid.token",
            "invalid.token.format",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
        ]

        for invalid_token in invalid_tokens:
            with pytest.raises(InvalidToken):
                jwt_auth.verify_token(invalid_token)

    def test_verify_expired_token(self, jwt_auth):
        """测试验证过期令牌"""
        # 创建一个已过期的认证器，使用相同的密钥
        expired_auth = JWTAuthenticator(
            secret_key=jwt_auth.secret_key,
            access_token_expire_minutes=-1  # 已过期
        )

        user_data = {"user_id": "test_user", "username": "testuser"}
        token = expired_auth.generate_access_token(user_data)

        with pytest.raises(TokenExpired):
            jwt_auth.verify_token(token)

    def test_refresh_access_token(self, jwt_auth):
        """测试刷新访问令牌"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com"
        }

        # 生成初始令牌对
        initial_tokens = jwt_auth.generate_token_pair(user_data)
        refresh_token = initial_tokens["refresh_token"]

        # 使用刷新令牌获取新的访问令牌
        new_tokens = jwt_auth.refresh_access_token(refresh_token)

        assert "access_token" in new_tokens
        assert "refresh_token" in new_tokens
        assert "token_type" in new_tokens
        assert "expires_in" in new_tokens

        # 验证新令牌有效
        new_payload = jwt_auth.verify_token(new_tokens["access_token"])
        assert new_payload["user_id"] == "test_user_123"
        assert new_payload["username"] == "testuser"

    def test_refresh_with_invalid_refresh_token(self, jwt_auth):
        """测试使用无效刷新令牌刷新"""
        invalid_refresh_tokens = [
            "",
            "invalid.token"
        ]

        for invalid_token in invalid_refresh_tokens:
            with pytest.raises(AuthenticationError):
                jwt_auth.refresh_access_token(invalid_token)

        # 使用访问令牌而非刷新令牌应该抛出认证错误
        access_token = jwt_auth.generate_access_token({"user_id": "test", "username": "test"})
        with pytest.raises(AuthenticationError):
            jwt_auth.refresh_access_token(access_token)

    def test_extract_token_from_header(self, jwt_auth):
        """测试从HTTP头部提取令牌"""
        # 正常的Bearer token
        header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
        token = jwt_auth.extract_token_from_header(header)
        assert token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"

        # 没有Bearer前缀
        header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
        token = jwt_auth.extract_token_from_header(header)
        assert token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"

        # 空字符串
        assert jwt_auth.extract_token_from_header("") is None

        # None值
        assert jwt_auth.extract_token_from_header(None) is None

    def test_authenticate_user(self, jwt_auth):
        """测试用户认证"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        # 生成令牌
        token = jwt_auth.generate_access_token(user_data)

        # 认证用户
        auth_result = jwt_auth.authenticate_user(token)

        assert auth_result is not None
        assert auth_result["user_id"] == "test_user_123"
        assert auth_result["username"] == "testuser"
        assert auth_result["email"] == "test@example.com"
        assert auth_result["role"] == "user"

    def test_authenticate_user_with_invalid_token(self, jwt_auth):
        """测试使用无效令牌认证用户"""
        with pytest.raises(AuthenticationError):
            jwt_auth.authenticate_user("invalid_token")

    def test_authenticate_user_with_expired_token(self, jwt_auth):
        """测试使用过期令牌认证用户"""
        expired_auth = JWTAuthenticator(
            secret_key="test_secret",
            access_token_expire_minutes=-1
        )

        user_data = {"user_id": "test_user", "username": "testuser"}
        token = expired_auth.generate_access_token(user_data)

        with pytest.raises(AuthenticationError):
            jwt_auth.authenticate_user(token)

    def test_get_token_remaining_time(self, jwt_auth):
        """测试获取令牌剩余时间"""
        user_data = {"user_id": "test_user", "username": "testuser"}
        token = jwt_auth.generate_access_token(user_data)

        remaining_time = jwt_auth.get_token_remaining_time(token)

        assert isinstance(remaining_time, int)
        assert remaining_time > 0
        assert remaining_time <= jwt_auth.access_token_expire_minutes * 60

    def test_get_token_remaining_time_for_invalid_token(self, jwt_auth):
        """测试获取无效令牌的剩余时间"""
        with pytest.raises(InvalidToken):
            jwt_auth.get_token_remaining_time("invalid_token")

    def test_is_token_expired(self, jwt_auth):
        """测试检查令牌是否过期"""
        user_data = {"user_id": "test_user", "username": "testuser"}
        valid_token = jwt_auth.generate_access_token(user_data)

        assert not jwt_auth.is_token_expired(valid_token)

        # 测试过期令牌
        expired_auth = JWTAuthenticator(
            secret_key="test_secret",
            access_token_expire_minutes=-1
        )
        expired_token = expired_auth.generate_access_token(user_data)

        assert jwt_auth.is_token_expired(expired_token)

    def test_revoke_token(self, jwt_auth):
        """测试撤销令牌"""
        user_data = {"user_id": "test_user", "username": "testuser"}
        token = jwt_auth.generate_access_token(user_data)

        # 令牌应该有效
        assert not jwt_auth.is_token_revoked(token)

        # 撤销令牌
        jwt_auth.revoke_token(token)

        # 令牌应该被撤销
        assert jwt_auth.is_token_revoked(token)

        # 验证被撤销的令牌应该失败
        with pytest.raises(InvalidToken):
            jwt_auth.verify_token(token)

    def test_revoke_multiple_tokens(self, jwt_auth):
        """测试撤销多个令牌"""
        user_data = {"user_id": "test_user", "username": "testuser"}

        # 生成多个令牌
        tokens = [jwt_auth.generate_access_token(user_data) for _ in range(5)]

        # 撤销所有令牌
        for token in tokens:
            jwt_auth.revoke_token(token)

        # 所有令牌都应该被撤销
        for token in tokens:
            assert jwt_auth.is_token_revoked(token)

    def test_cleanup_expired_tokens(self, jwt_auth):
        """测试清理过期令牌"""
        user_data = {"user_id": "test_user", "username": "testuser"}

        # 生成一些令牌
        tokens = [jwt_auth.generate_access_token(user_data) for _ in range(3)]

        # 撤销一些令牌
        jwt_auth.revoke_token(tokens[0])
        jwt_auth.revoke_token(tokens[1])

        # 清理过期令牌（模拟时间流逝）
        initial_count = jwt_auth.get_revoked_tokens_count()
        cleaned_count = jwt_auth.cleanup_expired_tokens()

        # 由于令牌还没有真正过期，数量应该不变
        assert jwt_auth.get_revoked_tokens_count() == initial_count
        assert cleaned_count == 0

    def test_different_secret_keys(self):
        """测试不同密钥生成的令牌"""
        auth1 = JWTAuthenticator(secret_key="secret1")
        auth2 = JWTAuthenticator(secret_key="secret2")

        user_data = {"user_id": "test_user", "username": "testuser"}

        token1 = auth1.generate_access_token(user_data)
        token2 = auth2.generate_access_token(user_data)

        # 令牌应该不同
        assert token1 != token2

        # 使用错误的密钥验证应该失败
        with pytest.raises(InvalidToken):
            auth2.verify_token(token1)

    def test_token_with_additional_claims(self, jwt_auth):
        """测试包含额外声明的令牌"""
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser",
            "permissions": ["read", "write"],
            "department": "engineering",
            "is_admin": False
        }

        token = jwt_auth.generate_access_token(user_data)
        payload = jwt_auth.verify_token(token)

        assert payload["user_id"] == "test_user_123"
        assert payload["username"] == "testuser"
        assert payload["permissions"] == ["read", "write"]
        assert payload["department"] == "engineering"
        assert payload["is_admin"] is False

    def test_performance_token_generation(self, jwt_auth):
        """测试令牌生成性能"""
        import time

        user_data = {"user_id": "test_user", "username": "testuser"}

        # 测试生成1000个令牌的时间
        start_time = time.time()
        for _ in range(1000):
            jwt_auth.generate_access_token(user_data)
        end_time = time.time()

        generation_time = end_time - start_time
        assert generation_time < 1.0  # 应该在1秒内完成

    def test_performance_token_verification(self, jwt_auth):
        """测试令牌验证性能"""
        import time

        user_data = {"user_id": "test_user", "username": "testuser"}
        token = jwt_auth.generate_access_token(user_data)

        # 测试验证1000次的时间
        start_time = time.time()
        for _ in range(1000):
            jwt_auth.verify_token(token)
        end_time = time.time()

        verification_time = end_time - start_time
        assert verification_time < 1.0  # 应该在1秒内完成