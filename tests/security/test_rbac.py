#!/usr/bin/env python3
"""
RBAC (基于角色的访问控制) 系统测试
测试角色、权限和用户的访问控制管理功能
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.rbac import (
    RBACManager, Role, Permission, User,
    RoleNotFoundError, PermissionNotFoundError, UserNotFoundError,
    DuplicateRoleError, DuplicatePermissionError, DuplicateUserError,
    InsufficientPermissionsError
)


class TestPermission:
    """权限测试类"""

    def test_permission_creation(self):
        """测试权限创建"""
        permission = Permission("read:users", "读取用户信息")

        assert permission.name == "read:users"
        assert permission.description == "读取用户信息"
        assert permission.resource == "users"
        assert permission.action == "read"
        assert str(permission) == "read:users"

    def test_permission_creation_with_resource_action(self):
        """测试通过资源和动作创建权限"""
        permission = Permission.create_from_resource_action("users", "read", "读取用户")

        assert permission.name == "read:users"
        assert permission.description == "读取用户"
        assert permission.resource == "users"
        assert permission.action == "read"

    def test_permission_equality(self):
        """测试权限相等性"""
        perm1 = Permission("read:users", "读取用户")
        perm2 = Permission("read:users", "读取用户信息")
        perm3 = Permission("write:users", "写入用户")

        assert perm1 == perm2
        assert perm1 != perm3
        assert hash(perm1) == hash(perm2)
        assert hash(perm1) != hash(perm3)

    def test_permission_wildcard_resource(self):
        """测试通配符资源权限"""
        permission = Permission("read:*", "读取所有资源")

        assert permission.resource == "*"
        assert permission.action == "read"
        assert permission.matches_resource("users")
        assert permission.matches_resource("posts")
        assert permission.matches_resource("*")

    def test_permission_wildcard_action(self):
        """测试通配符动作权限"""
        permission = Permission("*:users", "用户所有操作")

        assert permission.resource == "users"
        assert permission.action == "*"
        assert permission.matches_action("read")
        assert permission.matches_action("write")
        assert permission.matches_action("*")

    def test_permission_double_wildcard(self):
        """测试双通配符权限"""
        permission = Permission("*:*", "所有权限")

        assert permission.resource == "*"
        assert permission.action == "*"
        assert permission.matches_resource("anything")
        assert permission.matches_action("anything")

    def test_permission_matches(self):
        """测试权限匹配"""
        # 精确匹配
        perm = Permission("read:users", "读取用户")
        assert perm.matches("read:users")
        assert not perm.matches("write:users")
        assert not perm.matches("read:posts")

        # 通配符匹配
        wildcard_perm = Permission("read:*", "读取所有")
        assert wildcard_perm.matches("read:users")
        assert wildcard_perm.matches("read:posts")
        assert not wildcard_perm.matches("write:users")

    def test_permission_serialization(self):
        """测试权限序列化"""
        permission = Permission("read:users", "读取用户")
        data = permission.to_dict()

        assert data["name"] == "read:users"
        assert data["description"] == "读取用户"
        assert data["resource"] == "users"
        assert data["action"] == "read"

        # 测试从字典恢复
        restored = Permission.from_dict(data)
        assert restored.name == permission.name
        assert restored.description == permission.description
        assert restored.resource == permission.resource
        assert restored.action == permission.action


class TestRole:
    """角色测试类"""

    def test_role_creation(self):
        """测试角色创建"""
        role = Role("admin", "系统管理员")

        assert role.name == "admin"
        assert role.description == "系统管理员"
        assert len(role.permissions) == 0
        assert str(role) == "admin"

    def test_role_equality(self):
        """测试角色相等性"""
        role1 = Role("admin", "管理员")
        role2 = Role("admin", "系统管理员")
        role3 = Role("user", "普通用户")

        assert role1 == role2
        assert role1 != role3
        assert hash(role1) == hash(role2)
        assert hash(role1) != hash(role3)

    def test_add_permission(self):
        """测试添加权限"""
        role = Role("admin", "管理员")
        perm = Permission("read:users", "读取用户")

        role.add_permission(perm)
        assert perm in role.permissions
        assert len(role.permissions) == 1

    def test_add_duplicate_permission(self):
        """测试添加重复权限"""
        role = Role("admin", "管理员")
        perm1 = Permission("read:users", "读取用户")
        perm2 = Permission("read:users", "读取用户信息")

        role.add_permission(perm1)
        role.add_permission(perm2)  # 重复权限

        assert len(role.permissions) == 1
        assert perm1 in role.permissions

    def test_remove_permission(self):
        """测试移除权限"""
        role = Role("admin", "管理员")
        perm = Permission("read:users", "读取用户")

        role.add_permission(perm)
        assert perm in role.permissions

        role.remove_permission(perm)
        assert perm not in role.permissions
        assert len(role.permissions) == 0

    def test_has_permission(self):
        """测试权限检查"""
        role = Role("admin", "管理员")
        read_perm = Permission("read:users", "读取用户")
        write_perm = Permission("write:users", "写入用户")

        role.add_permission(read_perm)

        assert role.has_permission("read:users")
        assert not role.has_permission("write:users")

        # 测试通配符权限
        wildcard_perm = Permission("*:users", "用户所有操作")
        role.add_permission(wildcard_perm)

        assert role.has_permission("read:users")
        assert role.has_permission("write:users")
        assert role.has_permission("delete:users")

    def test_get_permissions(self):
        """测试获取权限列表"""
        role = Role("admin", "管理员")
        perm1 = Permission("read:users", "读取用户")
        perm2 = Permission("write:users", "写入用户")

        role.add_permission(perm1)
        role.add_permission(perm2)

        permissions = role.get_permissions()
        assert len(permissions) == 2
        assert perm1 in permissions
        assert perm2 in permissions

    def test_role_serialization(self):
        """测试角色序列化"""
        role = Role("admin", "管理员")
        perm = Permission("read:users", "读取用户")
        role.add_permission(perm)

        data = role.to_dict()
        assert data["name"] == "admin"
        assert data["description"] == "管理员"
        assert len(data["permissions"]) == 1
        assert data["permissions"][0]["name"] == "read:users"

        # 测试从字典恢复
        restored = Role.from_dict(data)
        assert restored.name == role.name
        assert restored.description == role.description
        assert len(restored.permissions) == 1
        assert restored.permissions[0].name == perm.name


class TestUser:
    """用户测试类"""

    def test_user_creation(self):
        """测试用户创建"""
        user = User("testuser", "test@example.com")

        assert user.user_id == "testuser"
        assert user.email == "test@example.com"
        assert len(user.roles) == 0
        assert user.is_active is True

    def test_user_creation_with_additional_fields(self):
        """测试带额外字段的用户创建"""
        user = User(
            user_id="testuser",
            email="test@example.com",
            full_name="Test User",
            department="IT"
        )

        assert user.user_id == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.department == "IT"

    def test_user_equality(self):
        """用户相等性测试"""
        user1 = User("testuser", "test@example.com")
        user2 = User("testuser", "test2@example.com")
        user3 = User("testuser2", "test@example.com")

        assert user1 == user2
        assert user1 != user3
        assert hash(user1) == hash(user2)
        assert hash(user1) != hash(user3)

    def test_add_role(self):
        """测试添加角色"""
        user = User("testuser", "test@example.com")
        role = Role("admin", "管理员")

        user.add_role(role)
        assert role in user.roles
        assert len(user.roles) == 1

    def test_add_duplicate_role(self):
        """测试添加重复角色"""
        user = User("testuser", "test@example.com")
        role1 = Role("admin", "管理员")
        role2 = Role("admin", "系统管理员")

        user.add_role(role1)
        user.add_role(role2)  # 重复角色

        assert len(user.roles) == 1
        assert role1 in user.roles

    def test_remove_role(self):
        """测试移除角色"""
        user = User("testuser", "test@example.com")
        role = Role("admin", "管理员")

        user.add_role(role)
        assert role in user.roles

        user.remove_role(role)
        assert role not in user.roles
        assert len(user.roles) == 0

    def test_has_role(self):
        """测试角色检查"""
        user = User("testuser", "test@example.com")
        admin_role = Role("admin", "管理员")
        user_role = Role("user", "普通用户")

        user.add_role(admin_role)

        assert user.has_role("admin")
        assert not user.has_role("user")

    def test_has_permission_via_role(self):
        """测试通过角色检查权限"""
        user = User("testuser", "test@example.com")
        admin_role = Role("admin", "管理员")
        read_perm = Permission("read:users", "读取用户")
        write_perm = Permission("write:users", "写入用户")

        admin_role.add_permission(read_perm)
        user.add_role(admin_role)

        assert user.has_permission("read:users")
        assert not user.has_permission("write:users")

        # 添加另一个角色的权限
        user_role = Role("editor", "编辑")
        user_role.add_permission(write_perm)
        user.add_role(user_role)

        assert user.has_permission("write:users")

    def test_get_all_permissions(self):
        """测试获取所有权限"""
        user = User("testuser", "test@example.com")
        admin_role = Role("admin", "管理员")
        user_role = Role("user", "普通用户")

        admin_role.add_permission(Permission("read:users", "读取用户"))
        admin_role.add_permission(Permission("write:users", "写入用户"))
        user_role.add_permission(Permission("read:posts", "读取文章"))

        user.add_role(admin_role)
        user.add_role(user_role)

        all_permissions = user.get_all_permissions()
        assert len(all_permissions) == 3
        permission_names = {perm.name for perm in all_permissions}
        assert "read:users" in permission_names
        assert "write:users" in permission_names
        assert "read:posts" in permission_names

    def test_activate_deactivate(self):
        """测试用户激活和停用"""
        user = User("testuser", "test@example.com")

        assert user.is_active is True

        user.deactivate()
        assert user.is_active is False

        user.activate()
        assert user.is_active is True

    def test_user_serialization(self):
        """测试用户序列化"""
        user = User(
            user_id="testuser",
            email="test@example.com",
            full_name="Test User",
            department="IT"
        )
        role = Role("admin", "管理员")
        perm = Permission("read:users", "读取用户")
        role.add_permission(perm)
        user.add_role(role)

        data = user.to_dict()
        assert data["user_id"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"
        assert data["department"] == "IT"
        assert data["is_active"] is True
        assert len(data["roles"]) == 1
        assert data["roles"][0]["name"] == "admin"

        # 测试从字典恢复
        restored = User.from_dict(data)
        assert restored.user_id == user.user_id
        assert restored.email == user.email
        assert restored.full_name == user.full_name
        assert restored.department == user.department
        assert len(restored.roles) == 1
        assert restored.roles[0].name == role.name


class TestRBACManager:
    """RBAC管理器测试类"""

    @pytest.fixture
    def rbac_manager(self):
        """创建RBAC管理器实例"""
        return RBACManager()

    def test_rbac_manager_initialization(self, rbac_manager):
        """测试RBAC管理器初始化"""
        assert len(rbac_manager.roles) == 0
        assert len(rbac_manager.permissions) == 0
        assert len(rbac_manager.users) == 0

    def test_create_permission(self, rbac_manager):
        """测试创建权限"""
        perm = rbac_manager.create_permission("read:users", "读取用户信息")

        assert isinstance(perm, Permission)
        assert perm.name == "read:users"
        assert perm.description == "读取用户信息"
        assert perm in rbac_manager.permissions

    def test_create_duplicate_permission(self, rbac_manager):
        """测试创建重复权限"""
        rbac_manager.create_permission("read:users", "读取用户信息")

        with pytest.raises(DuplicatePermissionError):
            rbac_manager.create_permission("read:users", "重复权限")

    def test_get_permission(self, rbac_manager):
        """测试获取权限"""
        perm = rbac_manager.create_permission("read:users", "读取用户信息")
        retrieved = rbac_manager.get_permission("read:users")

        assert retrieved == perm
        assert retrieved.description == "读取用户信息"

    def test_get_nonexistent_permission(self, rbac_manager):
        """测试获取不存在的权限"""
        with pytest.raises(PermissionNotFoundError):
            rbac_manager.get_permission("nonexistent:perm")

    def test_delete_permission(self, rbac_manager):
        """测试删除权限"""
        perm = rbac_manager.create_permission("read:users", "读取用户信息")
        assert perm in rbac_manager.permissions

        rbac_manager.delete_permission("read:users")
        assert perm not in rbac_manager.permissions

    def test_delete_nonexistent_permission(self, rbac_manager):
        """测试删除不存在的权限"""
        with pytest.raises(PermissionNotFoundError):
            rbac_manager.delete_permission("nonexistent:perm")

    def test_create_role(self, rbac_manager):
        """测试创建角色"""
        role = rbac_manager.create_role("admin", "系统管理员")

        assert isinstance(role, Role)
        assert role.name == "admin"
        assert role.description == "系统管理员"
        assert role in rbac_manager.roles

    def test_create_duplicate_role(self, rbac_manager):
        """测试创建重复角色"""
        rbac_manager.create_role("admin", "管理员")

        with pytest.raises(DuplicateRoleError):
            rbac_manager.create_role("admin", "重复管理员")

    def test_get_role(self, rbac_manager):
        """测试获取角色"""
        role = rbac_manager.create_role("admin", "系统管理员")
        retrieved = rbac_manager.get_role("admin")

        assert retrieved == role
        assert retrieved.description == "系统管理员"

    def test_get_nonexistent_role(self, rbac_manager):
        """测试获取不存在的角色"""
        with pytest.raises(RoleNotFoundError):
            rbac_manager.get_role("nonexistent_role")

    def test_delete_role(self, rbac_manager):
        """测试删除角色"""
        role = rbac_manager.create_role("admin", "系统管理员")
        assert role in rbac_manager.roles

        rbac_manager.delete_role("admin")
        assert role not in rbac_manager.roles

    def test_delete_nonexistent_role(self, rbac_manager):
        """测试删除不存在的角色"""
        with pytest.raises(RoleNotFoundError):
            rbac_manager.delete_role("nonexistent_role")

    def test_assign_permission_to_role(self, rbac_manager):
        """测试为角色分配权限"""
        perm = rbac_manager.create_permission("read:users", "读取用户")
        role = rbac_manager.create_role("admin", "管理员")

        rbac_manager.assign_permission_to_role("admin", "read:users")

        assert role.has_permission("read:users")
        assert perm in role.permissions

    def test_assign_permission_to_nonexistent_role(self, rbac_manager):
        """测试为不存在的角色分配权限"""
        rbac_manager.create_permission("read:users", "读取用户")

        with pytest.raises(RoleNotFoundError):
            rbac_manager.assign_permission_to_role("nonexistent", "read:users")

    def test_assign_nonexistent_permission_to_role(self, rbac_manager):
        """测试为角色分配不存在的权限"""
        rbac_manager.create_role("admin", "管理员")

        with pytest.raises(PermissionNotFoundError):
            rbac_manager.assign_permission_to_role("admin", "nonexistent:perm")

    def test_revoke_permission_from_role(self, rbac_manager):
        """测试撤销角色权限"""
        perm = rbac_manager.create_permission("read:users", "读取用户")
        role = rbac_manager.create_role("admin", "管理员")

        rbac_manager.assign_permission_to_role("admin", "read:users")
        assert role.has_permission("read:users")

        rbac_manager.revoke_permission_from_role("admin", "read:users")
        assert not role.has_permission("read:users")

    def test_create_user(self, rbac_manager):
        """测试创建用户"""
        user = rbac_manager.create_user("testuser", "test@example.com")

        assert isinstance(user, User)
        assert user.user_id == "testuser"
        assert user.email == "test@example.com"
        assert user in rbac_manager.users

    def test_create_duplicate_user(self, rbac_manager):
        """测试创建重复用户"""
        rbac_manager.create_user("testuser", "test@example.com")

        with pytest.raises(DuplicateUserError):
            rbac_manager.create_user("testuser", "test2@example.com")

    def test_get_user(self, rbac_manager):
        """测试获取用户"""
        user = rbac_manager.create_user("testuser", "test@example.com")
        retrieved = rbac_manager.get_user("testuser")

        assert retrieved == user
        assert retrieved.email == "test@example.com"

    def test_get_nonexistent_user(self, rbac_manager):
        """测试获取不存在的用户"""
        with pytest.raises(UserNotFoundError):
            rbac_manager.get_user("nonexistent_user")

    def test_delete_user(self, rbac_manager):
        """测试删除用户"""
        user = rbac_manager.create_user("testuser", "test@example.com")
        assert user in rbac_manager.users

        rbac_manager.delete_user("testuser")
        assert user not in rbac_manager.users

    def test_delete_nonexistent_user(self, rbac_manager):
        """测试删除不存在的用户"""
        with pytest.raises(UserNotFoundError):
            rbac_manager.delete_user("nonexistent_user")

    def test_assign_role_to_user(self, rbac_manager):
        """测试为用户分配角色"""
        role = rbac_manager.create_role("admin", "管理员")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_role_to_user("testuser", "admin")

        assert user.has_role("admin")
        assert role in user.roles

    def test_assign_role_to_nonexistent_user(self, rbac_manager):
        """测试为不存在的用户分配角色"""
        rbac_manager.create_role("admin", "管理员")

        with pytest.raises(UserNotFoundError):
            rbac_manager.assign_role_to_user("nonexistent", "admin")

    def test_assign_nonexistent_role_to_user(self, rbac_manager):
        """测试为用户分配不存在的角色"""
        rbac_manager.create_user("testuser", "test@example.com")

        with pytest.raises(RoleNotFoundError):
            rbac_manager.assign_role_to_user("testuser", "nonexistent")

    def test_revoke_role_from_user(self, rbac_manager):
        """测试撤销用户角色"""
        role = rbac_manager.create_role("admin", "管理员")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_role_to_user("testuser", "admin")
        assert user.has_role("admin")

        rbac_manager.revoke_role_from_user("testuser", "admin")
        assert not user.has_role("admin")

    def test_check_user_permission(self, rbac_manager):
        """测试检查用户权限"""
        # 设置权限和角色
        perm = rbac_manager.create_permission("read:users", "读取用户")
        role = rbac_manager.create_role("admin", "管理员")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_permission_to_role("admin", "read:users")
        rbac_manager.assign_role_to_user("testuser", "admin")

        # 检查权限
        assert rbac_manager.check_user_permission("testuser", "read:users")
        assert not rbac_manager.check_user_permission("testuser", "write:users")

    def test_check_user_permission_with_wildcard(self, rbac_manager):
        """测试通配符权限检查"""
        # 设置通配符权限
        rbac_manager.create_permission("*:users", "用户所有操作")
        rbac_manager.create_role("admin", "管理员")
        rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_permission_to_role("admin", "*:users")
        rbac_manager.assign_role_to_user("testuser", "admin")

        # 检查各种权限
        assert rbac_manager.check_user_permission("testuser", "read:users")
        assert rbac_manager.check_user_permission("testuser", "write:users")
        assert rbac_manager.check_user_permission("testuser", "delete:users")
        assert not rbac_manager.check_user_permission("testuser", "read:posts")

    def test_check_inactive_user_permission(self, rbac_manager):
        """测试非活跃用户权限检查"""
        perm = rbac_manager.create_permission("read:users", "读取用户")
        role = rbac_manager.create_role("admin", "管理员")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_permission_to_role("admin", "read:users")
        rbac_manager.assign_role_to_user("testuser", "admin")

        # 停用用户
        user.deactivate()

        # 非活跃用户不应该有任何权限
        assert not rbac_manager.check_user_permission("testuser", "read:users")

    def test_require_permission_success(self, rbac_manager):
        """测试权限要求成功"""
        perm = rbac_manager.create_permission("read:users", "读取用户")
        role = rbac_manager.create_role("admin", "管理员")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_permission_to_role("admin", "read:users")
        rbac_manager.assign_role_to_user("testuser", "admin")

        # 应该不抛出异常
        rbac_manager.require_permission("testuser", "read:users")

    def test_require_permission_failure(self, rbac_manager):
        """测试权限要求失败"""
        rbac_manager.create_user("testuser", "test@example.com")

        with pytest.raises(InsufficientPermissionsError):
            rbac_manager.require_permission("testuser", "nonexistent:perm")

    def test_get_user_permissions(self, rbac_manager):
        """测试获取用户所有权限"""
        # 设置多个角色和权限
        rbac_manager.create_permission("read:users", "读取用户")
        rbac_manager.create_permission("write:users", "写入用户")
        rbac_manager.create_permission("read:posts", "读取文章")

        admin_role = rbac_manager.create_role("admin", "管理员")
        editor_role = rbac_manager.create_role("editor", "编辑")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_permission_to_role("admin", "read:users")
        rbac_manager.assign_permission_to_role("admin", "write:users")
        rbac_manager.assign_permission_to_role("editor", "read:posts")

        rbac_manager.assign_role_to_user("testuser", "admin")
        rbac_manager.assign_role_to_user("testuser", "editor")

        user_permissions = rbac_manager.get_user_permissions("testuser")
        permission_names = {perm.name for perm in user_permissions}

        assert len(user_permissions) == 3
        assert "read:users" in permission_names
        assert "write:users" in permission_names
        assert "read:posts" in permission_names

    def test_get_user_roles(self, rbac_manager):
        """测试获取用户角色"""
        admin_role = rbac_manager.create_role("admin", "管理员")
        user_role = rbac_manager.create_role("user", "普通用户")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_role_to_user("testuser", "admin")
        rbac_manager.assign_role_to_user("testuser", "user")

        user_roles = rbac_manager.get_user_roles("testuser")
        role_names = {role.name for role in user_roles}

        assert len(user_roles) == 2
        assert "admin" in role_names
        assert "user" in role_names

    def test_list_all_permissions(self, rbac_manager):
        """测试列出所有权限"""
        rbac_manager.create_permission("read:users", "读取用户")
        rbac_manager.create_permission("write:users", "写入用户")
        rbac_manager.create_permission("read:posts", "读取文章")

        all_permissions = rbac_manager.list_all_permissions()
        permission_names = {perm.name for perm in all_permissions}

        assert len(all_permissions) == 3
        assert "read:users" in permission_names
        assert "write:users" in permission_names
        assert "read:posts" in permission_names

    def test_list_all_roles(self, rbac_manager):
        """测试列出所有角色"""
        rbac_manager.create_role("admin", "管理员")
        rbac_manager.create_role("user", "普通用户")
        rbac_manager.create_role("editor", "编辑")

        all_roles = rbac_manager.list_all_roles()
        role_names = {role.name for role in all_roles}

        assert len(all_roles) == 3
        assert "admin" in role_names
        assert "user" in role_names
        assert "editor" in role_names

    def test_list_all_users(self, rbac_manager):
        """测试列出所有用户"""
        rbac_manager.create_user("user1", "user1@example.com")
        rbac_manager.create_user("user2", "user2@example.com")
        rbac_manager.create_user("user3", "user3@example.com")

        all_users = rbac_manager.list_all_users()
        user_ids = {user.user_id for user in all_users}

        assert len(all_users) == 3
        assert "user1" in user_ids
        assert "user2" in user_ids
        assert "user3" in user_ids

    def test_cascade_delete_role_removes_from_users(self, rbac_manager):
        """测试级联删除角色时从用户中移除"""
        role = rbac_manager.create_role("admin", "管理员")
        user = rbac_manager.create_user("testuser", "test@example.com")

        rbac_manager.assign_role_to_user("testuser", "admin")
        assert user.has_role("admin")

        rbac_manager.delete_role("admin")
        assert not user.has_role("admin")
        assert role not in rbac_manager.roles

    def test_role_hierarchy_check(self, rbac_manager):
        """测试角色层次检查"""
        # 创建层次角色
        admin_role = rbac_manager.create_role("admin", "管理员")
        manager_role = rbac_manager.create_role("manager", "经理")
        user_role = rbac_manager.create_role("user", "普通用户")

        # 创建权限
        read_perm = rbac_manager.create_permission("read:data", "读取数据")
        write_perm = rbac_manager.create_permission("write:data", "写入数据")
        admin_perm = rbac_manager.create_permission("admin:system", "系统管理")

        # 分配权限
        rbac_manager.assign_permission_to_role("user", "read:data")
        rbac_manager.assign_permission_to_role("manager", "write:data")
        rbac_manager.assign_permission_to_role("admin", "admin:system")

        # 创建用户并分配多个角色
        user = rbac_manager.create_user("testuser", "test@example.com")
        rbac_manager.assign_role_to_user("testuser", "user")
        rbac_manager.assign_role_to_user("testuser", "manager")
        rbac_manager.assign_role_to_user("testuser", "admin")

        # 检查用户拥有所有角色的权限
        assert rbac_manager.check_user_permission("testuser", "read:data")
        assert rbac_manager.check_user_permission("testuser", "write:data")
        assert rbac_manager.check_user_permission("testuser", "admin:system")

    def test_performance_large_scale(self, rbac_manager):
        """测试大规模性能"""
        import time

        # 创建大量权限、角色和用户
        start_time = time.time()

        # 创建100个权限
        for i in range(100):
            rbac_manager.create_permission(f"action{i}:resource{i}", f"权限{i}")

        # 创建10个角色
        roles = []
        for i in range(10):
            role = rbac_manager.create_role(f"role{i}", f"角色{i}")
            roles.append(role)

            # 每个角色分配10个权限
            for j in range(10):
                perm_name = f"action{j}:resource{j}"
                rbac_manager.assign_permission_to_role(f"role{i}", perm_name)

        # 创建100个用户
        for i in range(100):
            user = rbac_manager.create_user(f"user{i}", f"user{i}@example.com")

            # 每个用户分配2-3个角色
            for j in range(i % 3 + 2):
                role_name = f"role{j}"
                rbac_manager.assign_role_to_user(f"user{i}", role_name)

        creation_time = time.time() - start_time

        # 测试权限检查性能
        start_time = time.time()
        for i in range(1000):
            user_id = f"user{i % 100}"
            perm_name = f"action{i % 10}:resource{i % 10}"
            rbac_manager.check_user_permission(user_id, perm_name)

        check_time = time.time() - start_time

        # 性能断言
        assert creation_time < 5.0  # 创建应该在5秒内完成
        assert check_time < 2.0    # 1000次权限检查应该在2秒内完成

    def test_export_import_data(self, rbac_manager):
        """测试数据导出导入"""
        # 创建测试数据
        rbac_manager.create_permission("read:users", "读取用户")
        rbac_manager.create_permission("write:users", "写入用户")

        rbac_manager.create_role("admin", "管理员")
        rbac_manager.create_role("user", "普通用户")

        rbac_manager.assign_permission_to_role("admin", "read:users")
        rbac_manager.assign_permission_to_role("admin", "write:users")
        rbac_manager.assign_permission_to_role("user", "read:users")

        user = rbac_manager.create_user("testuser", "test@example.com")
        rbac_manager.assign_role_to_user("testuser", "admin")

        # 导出数据
        exported_data = rbac_manager.export_data()

        # 验证导出数据结构
        assert "permissions" in exported_data
        assert "roles" in exported_data
        assert "users" in exported_data
        assert len(exported_data["permissions"]) == 2
        assert len(exported_data["roles"]) == 2
        assert len(exported_data["users"]) == 1

        # 创建新的管理器并导入数据
        new_manager = RBACManager()
        new_manager.import_data(exported_data)

        # 验证导入的数据
        assert len(new_manager.permissions) == 2
        assert len(new_manager.roles) == 2
        assert len(new_manager.users) == 1

        # 验证关系
        assert new_manager.check_user_permission("testuser", "read:users")
        assert new_manager.check_user_permission("testuser", "write:users")

        imported_user = new_manager.get_user("testuser")
        assert imported_user.has_role("admin")