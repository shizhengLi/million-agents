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
from .audit_log import (
    AuditEvent, AuditLog, AuditLogManager, AuditSeverity,
    AuditCategory, AuditEventType, AuditSearchFilter,
    AuditStatistics, AuditExporter, AuditImporter,
    create_audit_event, log_security_event
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
    "InsufficientPermissionsError",
    # 审计日志
    "AuditEvent",
    "AuditLog",
    "AuditLogManager",
    "AuditSeverity",
    "AuditCategory",
    "AuditEventType",
    "AuditSearchFilter",
    "AuditStatistics",
    "AuditExporter",
    "AuditImporter",
    "create_audit_event",
    "log_security_event"
]