"""
RBAC (基于角色的访问控制) 模块

提供完整的角色、权限和用户访问控制管理功能
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import logging


class RBACError(Exception):
    """RBAC相关错误的基类"""
    pass


class RoleNotFoundError(RBACError):
    """角色不存在异常"""
    pass


class PermissionNotFoundError(RBACError):
    """权限不存在异常"""
    pass


class UserNotFoundError(RBACError):
    """用户不存在异常"""
    pass


class DuplicateRoleError(RBACError):
    """重复角色异常"""
    pass


class DuplicatePermissionError(RBACError):
    """重复权限异常"""
    pass


class DuplicateUserError(RBACError):
    """重复用户异常"""
    pass


class InsufficientPermissionsError(RBACError):
    """权限不足异常"""
    pass


@dataclass
class Permission:
    """权限类"""
    name: str
    description: str
    resource: str = field(init=False)
    action: str = field(init=False)

    def __post_init__(self):
        """解析权限名称为资源和动作"""
        if ":" in self.name:
            parts = self.name.split(":", 1)
            self.resource = parts[1]
            self.action = parts[0]
        else:
            self.resource = self.name
            self.action = "read"

    @classmethod
    def create_from_resource_action(cls, resource: str, action: str, description: str) -> 'Permission':
        """从资源和动作创建权限"""
        return cls(f"{action}:{resource}", description)

    def matches(self, permission_name: str) -> bool:
        """检查权限是否匹配"""
        if self.name == permission_name:
            return True

        # 解析目标权限
        if ":" in permission_name:
            target_action, target_resource = permission_name.split(":", 1)
        else:
            target_action = "read"
            target_resource = permission_name

        # 检查通配符匹配
        return self.matches_action(target_action) and self.matches_resource(target_resource)

    def matches_action(self, action: str) -> bool:
        """检查动作是否匹配"""
        return self.action == "*" or self.action == action

    def matches_resource(self, resource: str) -> bool:
        """检查资源是否匹配"""
        return self.resource == "*" or self.resource == resource

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        """权限相等性比较"""
        if not isinstance(other, Permission):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        """权限哈希值"""
        return hash(self.name)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "resource": self.resource,
            "action": self.action
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """从字典创建权限"""
        return cls(data["name"], data["description"])


@dataclass
class Role:
    """角色类"""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)

    def add_permission(self, permission: Permission) -> None:
        """添加权限"""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission) -> None:
        """移除权限"""
        self.permissions.discard(permission)

    def has_permission(self, permission_name: str) -> bool:
        """检查是否拥有指定权限"""
        for permission in self.permissions:
            if permission.matches(permission_name):
                return True
        return False

    def get_permissions(self) -> List[Permission]:
        """获取所有权限列表"""
        return list(self.permissions)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        """角色相等性比较"""
        if not isinstance(other, Role):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        """角色哈希值"""
        return hash(self.name)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [perm.to_dict() for perm in self.permissions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """从字典创建角色"""
        role = cls(data["name"], data["description"])
        for perm_data in data.get("permissions", []):
            permission = Permission.from_dict(perm_data)
            role.add_permission(permission)
        return role


@dataclass
class User:
    """用户类"""
    user_id: str
    email: str
    full_name: Optional[str] = None
    department: Optional[str] = None
    roles: Set[Role] = field(default_factory=set)
    is_active: bool = True

    def add_role(self, role: Role) -> None:
        """添加角色"""
        self.roles.add(role)

    def remove_role(self, role: Role) -> None:
        """移除角色"""
        self.roles.discard(role)

    def has_role(self, role_name: str) -> bool:
        """检查是否拥有指定角色"""
        return any(role.name == role_name for role in self.roles)

    def has_permission(self, permission_name: str) -> bool:
        """通过角色检查是否拥有指定权限"""
        if not self.is_active:
            return False

        for role in self.roles:
            if role.has_permission(permission_name):
                return True
        return False

    def get_all_permissions(self) -> List[Permission]:
        """获取所有角色的权限"""
        all_permissions = set()
        for role in self.roles:
            all_permissions.update(role.permissions)
        return list(all_permissions)

    def activate(self) -> None:
        """激活用户"""
        self.is_active = True

    def deactivate(self) -> None:
        """停用用户"""
        self.is_active = False

    def __str__(self) -> str:
        return self.user_id

    def __eq__(self, other) -> bool:
        """用户相等性比较"""
        if not isinstance(other, User):
            return False
        return self.user_id == other.user_id

    def __hash__(self) -> int:
        """用户哈希值"""
        return hash(self.user_id)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "full_name": self.full_name,
            "department": self.department,
            "is_active": self.is_active,
            "roles": [role.to_dict() for role in self.roles]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """从字典创建用户"""
        user = cls(
            user_id=data["user_id"],
            email=data["email"],
            full_name=data.get("full_name"),
            department=data.get("department"),
            is_active=data.get("is_active", True)
        )
        for role_data in data.get("roles", []):
            role = Role.from_dict(role_data)
            user.add_role(role)
        return user


class RBACManager:
    """RBAC管理器"""

    def __init__(self):
        """初始化RBAC管理器"""
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.logger = logging.getLogger(__name__)

    # 权限管理
    def create_permission(self, name: str, description: str) -> Permission:
        """创建权限"""
        if name in self.permissions:
            raise DuplicatePermissionError(f"权限 '{name}' 已存在")

        permission = Permission(name, description)
        self.permissions[name] = permission
        self.logger.info(f"创建权限: {name}")
        return permission

    def get_permission(self, name: str) -> Permission:
        """获取权限"""
        if name not in self.permissions:
            raise PermissionNotFoundError(f"权限 '{name}' 不存在")
        return self.permissions[name]

    def delete_permission(self, name: str) -> None:
        """删除权限"""
        if name not in self.permissions:
            raise PermissionNotFoundError(f"权限 '{name}' 不存在")

        permission = self.permissions[name]

        # 从所有角色中移除此权限
        for role in self.roles.values():
            role.remove_permission(permission)

        del self.permissions[name]
        self.logger.info(f"删除权限: {name}")

    # 角色管理
    def create_role(self, name: str, description: str) -> Role:
        """创建角色"""
        if name in self.roles:
            raise DuplicateRoleError(f"角色 '{name}' 已存在")

        role = Role(name, description)
        self.roles[name] = role
        self.logger.info(f"创建角色: {name}")
        return role

    def get_role(self, name: str) -> Role:
        """获取角色"""
        if name not in self.roles:
            raise RoleNotFoundError(f"角色 '{name}' 不存在")
        return self.roles[name]

    def delete_role(self, name: str) -> None:
        """删除角色"""
        if name not in self.roles:
            raise RoleNotFoundError(f"角色 '{name}' 不存在")

        role = self.roles[name]

        # 从所有用户中移除此角色
        for user in self.users.values():
            user.remove_role(role)

        del self.roles[name]
        self.logger.info(f"删除角色: {name}")

    def assign_permission_to_role(self, role_name: str, permission_name: str) -> None:
        """为角色分配权限"""
        role = self.get_role(role_name)
        permission = self.get_permission(permission_name)

        role.add_permission(permission)
        self.logger.info(f"为角色 '{role_name}' 分配权限 '{permission_name}'")

    def revoke_permission_from_role(self, role_name: str, permission_name: str) -> None:
        """撤销角色权限"""
        role = self.get_role(role_name)
        permission = self.get_permission(permission_name)

        role.remove_permission(permission)
        self.logger.info(f"撤销角色 '{role_name}' 的权限 '{permission_name}'")

    # 用户管理
    def create_user(self, user_id: str, email: str, **kwargs) -> User:
        """创建用户"""
        if user_id in self.users:
            raise DuplicateUserError(f"用户 '{user_id}' 已存在")

        user = User(user_id=user_id, email=email, **kwargs)
        self.users[user_id] = user
        self.logger.info(f"创建用户: {user_id}")
        return user

    def get_user(self, user_id: str) -> User:
        """获取用户"""
        if user_id not in self.users:
            raise UserNotFoundError(f"用户 '{user_id}' 不存在")
        return self.users[user_id]

    def delete_user(self, user_id: str) -> None:
        """删除用户"""
        if user_id not in self.users:
            raise UserNotFoundError(f"用户 '{user_id}' 不存在")

        del self.users[user_id]
        self.logger.info(f"删除用户: {user_id}")

    def assign_role_to_user(self, user_id: str, role_name: str) -> None:
        """为用户分配角色"""
        user = self.get_user(user_id)
        role = self.get_role(role_name)

        user.add_role(role)
        self.logger.info(f"为用户 '{user_id}' 分配角色 '{role_name}'")

    def revoke_role_from_user(self, user_id: str, role_name: str) -> None:
        """撤销用户角色"""
        user = self.get_user(user_id)
        role = self.get_role(role_name)

        user.remove_role(role)
        self.logger.info(f"撤销用户 '{user_id}' 的角色 '{role_name}'")

    # 权限检查
    def check_user_permission(self, user_id: str, permission_name: str) -> bool:
        """检查用户权限"""
        try:
            user = self.get_user(user_id)
            return user.has_permission(permission_name)
        except UserNotFoundError:
            return False

    def require_permission(self, user_id: str, permission_name: str) -> None:
        """要求用户拥有指定权限"""
        if not self.check_user_permission(user_id, permission_name):
            raise InsufficientPermissionsError(
                f"用户 '{user_id}' 缺少权限 '{permission_name}'"
            )

    # 查询方法
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """获取用户所有权限"""
        user = self.get_user(user_id)
        return user.get_all_permissions()

    def get_user_roles(self, user_id: str) -> List[Role]:
        """获取用户所有角色"""
        user = self.get_user(user_id)
        return list(user.roles)

    def list_all_permissions(self) -> List[Permission]:
        """列出所有权限"""
        return list(self.permissions.values())

    def list_all_roles(self) -> List[Role]:
        """列出所有角色"""
        return list(self.roles.values())

    def list_all_users(self) -> List[User]:
        """列出所有用户"""
        return list(self.users.values())

    # 数据导出导入
    def export_data(self) -> Dict[str, Any]:
        """导出所有数据"""
        return {
            "permissions": [perm.to_dict() for perm in self.permissions.values()],
            "roles": [role.to_dict() for role in self.roles.values()],
            "users": [user.to_dict() for user in self.users.values()]
        }

    def import_data(self, data: Dict[str, Any]) -> None:
        """导入数据"""
        # 清空现有数据
        self.permissions.clear()
        self.roles.clear()
        self.users.clear()

        # 导入权限
        for perm_data in data.get("permissions", []):
            permission = Permission.from_dict(perm_data)
            self.permissions[permission.name] = permission

        # 导入角色
        for role_data in data.get("roles", []):
            role = Role.from_dict(role_data)
            self.roles[role.name] = role

        # 导入用户
        for user_data in data.get("users", []):
            user = User.from_dict(user_data)
            self.users[user.user_id] = user

        self.logger.info("数据导入完成")

    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "permissions_count": len(self.permissions),
            "roles_count": len(self.roles),
            "users_count": len(self.users),
            "active_users_count": len([u for u in self.users.values() if u.is_active])
        }