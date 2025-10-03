"""
安全模块

提供认证、授权、加密等安全功能
"""

from .jwt_auth import JWTAuthenticator, TokenExpired, InvalidToken, AuthenticationError
from .encryption import (
    DataEncryption, EncryptionError, KeyGenerationError,
    AESCipher, RSACipher, KeyManager
)
from .rbac import (
    RBACManager, Role, Permission, User,
    RoleNotFoundError, PermissionNotFoundError, UserNotFoundError,
    DuplicateRoleError, DuplicatePermissionError, DuplicateUserError,
    InsufficientPermissionsError
)

__all__ = [
    # JWT认证
    "JWTAuthenticator",
    "TokenExpired",
    "InvalidToken",
    "AuthenticationError",
    # 数据加密
    "DataEncryption",
    "EncryptionError",
    "KeyGenerationError",
    "AESCipher",
    "RSACipher",
    "KeyManager",
    # RBAC访问控制
    "RBACManager",
    "Role",
    "Permission",
    "User",
    "RoleNotFoundError",
    "PermissionNotFoundError",
    "UserNotFoundError",
    "DuplicateRoleError",
    "DuplicatePermissionError",
    "DuplicateUserError",
    "InsufficientPermissionsError"
]