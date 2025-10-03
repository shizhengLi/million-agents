"""
JWT认证模块

提供JWT token的生成、验证、刷新等功能
"""

import jwt
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass
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


class SecurityError(Exception):
    """安全相关错误的基类"""
    pass


class TokenExpired(SecurityError):
    """Token过期异常"""
    pass


class InvalidToken(SecurityError):
    """无效Token异常"""
    pass


class AuthenticationError(SecurityError):
    """认证错误异常"""
    pass


@dataclass
class TokenPayload:
    """JWT令牌载荷"""
    user_id: str
    username: str
    email: Optional[str] = None
    role: str = "user"
    permissions: list = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    type: str = "access"  # access or refresh
    jti: Optional[str] = None  # JWT ID

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


class JWTAuthenticator:
    """JWT认证器"""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        """初始化JWT认证器

        Args:
            secret_key: JWT签名密钥
            algorithm: 签名算法
            access_token_expire_minutes: 访问令牌过期时间（分钟）
            refresh_token_expire_days: 刷新令牌过期时间（天）
        """
        self.settings = Settings()
        self.secret_key = secret_key or self.settings.openai_api_key or "default_secret_key_change_in_production"
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

        # 令牌撤销管理
        self._revoked_tokens: Set[str] = set()
        self._revoked_tokens_lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    def _generate_jti(self) -> str:
        """生成JWT ID"""
        import uuid
        return str(uuid.uuid4())

    def _create_payload(self, user_data: Dict[str, Any], token_type: str = "access") -> Dict[str, Any]:
        """创建JWT载荷

        Args:
            user_data: 用户数据
            token_type: 令牌类型 (access/refresh)

        Returns:
            JWT载荷字典
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        if token_type == "access":
            expire_time = now + timedelta(minutes=self.access_token_expire_minutes)
        else:  # refresh
            expire_time = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "exp": int(expire_time.timestamp()),
            "iat": int(now.timestamp()),
            "type": token_type,
            "jti": self._generate_jti()
        }

        # 添加所有额外字段
        for field, value in user_data.items():
            if field not in ["user_id", "username"]:  # 跳过已处理的必需字段
                payload[field] = value

        # 默认角色
        if "role" not in payload:
            payload["role"] = "user"

        return payload

    def generate_access_token(self, user_data: Dict[str, Any]) -> str:
        """生成访问令牌

        Args:
            user_data: 用户数据，必须包含user_id和username

        Returns:
            JWT访问令牌字符串

        Raises:
            ValueError: 当用户数据缺少必需字段时
        """
        if "user_id" not in user_data or "username" not in user_data:
            raise ValueError("用户数据必须包含user_id和username字段")

        payload = self._create_payload(user_data, "access")
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        self.logger.info(f"为用户 {user_data['username']} 生成访问令牌")
        return token

    def generate_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """生成刷新令牌

        Args:
            user_data: 用户数据，必须包含user_id和username

        Returns:
            JWT刷新令牌字符串
        """
        if "user_id" not in user_data or "username" not in user_data:
            raise ValueError("用户数据必须包含user_id和username字段")

        payload = self._create_payload(user_data, "refresh")
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        self.logger.info(f"为用户 {user_data['username']} 生成刷新令牌")
        return token

    def generate_token_pair(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成令牌对（访问令牌和刷新令牌）

        Args:
            user_data: 用户数据

        Returns:
            包含令牌对的字典
        """
        access_token = self.generate_access_token(user_data)
        refresh_token = self.generate_refresh_token(user_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证JWT令牌

        Args:
            token: JWT令牌字符串

        Returns:
            解码后的载荷字典

        Raises:
            InvalidToken: 当令牌无效时
            TokenExpired: 当令牌过期时
        """
        if not token:
            raise InvalidToken("令牌不能为空")

        # 检查令牌是否被撤销
        if self.is_token_revoked(token):
            raise InvalidToken("令牌已被撤销")

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "iat", "type"]}
            )

            # 验证令牌类型
            if "type" not in payload:
                raise InvalidToken("令牌缺少类型字段")

            return payload

        except jwt.ExpiredSignatureError:
            raise TokenExpired("令牌已过期")
        except jwt.InvalidTokenError as e:
            raise InvalidToken(f"无效令牌: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """使用刷新令牌获取新的访问令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            包含新令牌对的字典

        Raises:
            InvalidToken: 当刷新令牌无效时
            AuthenticationError: 当认证失败时
        """
        try:
            # 验证刷新令牌
            payload = self.verify_token(refresh_token)

            if payload.get("type") != "refresh":
                raise InvalidToken("必须使用刷新令牌")

            # 提取用户数据
            user_data = {
                "user_id": payload["user_id"],
                "username": payload["username"]
            }

            # 添加所有额外字段
            for field, value in payload.items():
                if field in ["email", "role", "permissions", "department", "is_admin"]:
                    user_data[field] = value

            # 生成新的令牌对
            new_tokens = self.generate_token_pair(user_data)

            # 撤销旧的刷新令牌（在生成新令牌之后）
            self.revoke_token(refresh_token)

            self.logger.info(f"为用户 {payload['username']} 刷新访问令牌")
            return new_tokens

        except (InvalidToken, TokenExpired) as e:
            raise AuthenticationError(f"刷新令牌失败: {str(e)}")

    def extract_token_from_header(self, authorization_header: Optional[str]) -> Optional[str]:
        """从HTTP头部提取令牌

        Args:
            authorization_header: Authorization头部值

        Returns:
            提取的令牌字符串，如果无法提取则返回None
        """
        if not authorization_header:
            return None

        # 移除Bearer前缀
        if authorization_header.startswith("Bearer "):
            return authorization_header[7:].strip()

        return authorization_header.strip() or None

    def authenticate_user(self, token: str) -> Dict[str, Any]:
        """认证用户

        Args:
            token: JWT访问令牌

        Returns:
            用户信息字典

        Raises:
            AuthenticationError: 当认证失败时
        """
        try:
            payload = self.verify_token(token)

            if payload.get("type") != "access":
                raise AuthenticationError("必须使用访问令牌进行认证")

            # 返回用户信息
            user_info = {
                "user_id": payload["user_id"],
                "username": payload["username"],
                "role": payload.get("role", "user"),
                "exp": payload["exp"],
                "iat": payload["iat"]
            }

            # 添加可选字段
            for field in ["email", "permissions"]:
                if field in payload:
                    user_info[field] = payload[field]

            self.logger.info(f"用户认证成功: {payload['username']}")
            return user_info

        except (InvalidToken, TokenExpired) as e:
            self.logger.warning(f"用户认证失败: {str(e)}")
            raise AuthenticationError(f"认证失败: {str(e)}")

    def get_token_remaining_time(self, token: str) -> int:
        """获取令牌剩余时间（秒）

        Args:
            token: JWT令牌

        Returns:
            剩余时间（秒）

        Raises:
            InvalidToken: 当令牌无效时
        """
        payload = self.verify_token(token)
        now = int(time.time())
        remaining_time = payload["exp"] - now

        return max(0, remaining_time)

    def is_token_expired(self, token: str) -> bool:
        """检查令牌是否过期

        Args:
            token: JWT令牌

        Returns:
            True如果过期，False否则
        """
        try:
            remaining_time = self.get_token_remaining_time(token)
            return remaining_time <= 0
        except InvalidToken:
            return True

    def revoke_token(self, token: str) -> None:
        """撤销令牌

        Args:
            token: 要撤销的令牌
        """
        if not token:
            return

        try:
            # 验证令牌并获取JTI
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False}
            )

            jti = payload.get("jti")
            if jti:
                with self._revoked_tokens_lock:
                    self._revoked_tokens.add(jti)

                self.logger.info(f"令牌已撤销: {jti}")

        except jwt.InvalidTokenError:
            # 如果令牌格式无效，直接忽略
            pass

    def is_token_revoked(self, token: str) -> bool:
        """检查令牌是否被撤销

        Args:
            token: JWT令牌

        Returns:
            True如果被撤销，False否则
        """
        if not token:
            return False

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False}
            )

            jti = payload.get("jti")
            if jti:
                with self._revoked_tokens_lock:
                    return jti in self._revoked_tokens

            return False

        except jwt.InvalidTokenError:
            return False

    def cleanup_expired_tokens(self) -> int:
        """清理过期的撤销令牌

        Returns:
            清理的令牌数量
        """
        if not self._revoked_tokens:
            return 0

        current_time = int(time.time())
        expired_tokens = set()

        # 注意：这是一个简化的实现
        # 在实际生产环境中，应该存储令牌的过期时间以便高效清理
        with self._revoked_tokens_lock:
            # 这里我们保留所有撤销的令牌，因为无法仅从JTI判断过期时间
            # 在实际应用中，应该使用数据库或Redis存储令牌撤销信息
            pass

        return len(expired_tokens)

    def get_revoked_tokens_count(self) -> int:
        """获取撤销令牌的数量

        Returns:
            撤销令牌的数量
        """
        with self._revoked_tokens_lock:
            return len(self._revoked_tokens)