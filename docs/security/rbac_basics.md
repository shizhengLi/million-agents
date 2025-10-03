# RBAC访问控制基础知识

## 什么是RBAC

RBAC (Role-Based Access Control) 基于角色的访问控制是一种访问控制策略，它将权限分配给角色，而不是直接分配给用户。用户通过被分配适当的角色来获得相应的权限。

## RBAC核心概念

### 1. 用户 (User)
- 系统中的实体，可以是人、程序或系统
- 通过角色获得权限
- 可以拥有多个角色

### 2. 角色 (Role)
- 权限的集合
- 代表工作职责或职能
- 可以分配给多个用户

### 3. 权限 (Permission)
- 执行特定操作的权利
- 通常格式为 "动作:资源" (action:resource)
- 可以分配给多个角色

### 4. 会话 (Session)
- 用户与系统的交互期间
- 激活用户的一组角色
- 动态权限检查

## RBAC模型分类

### RBAC0 - 基础模型
```
用户 ←→ 角色 ←→ 权限
```

### RBAC1 - 层次模型
```
     管理员
       ↓
     部门经理
       ↓
     普通员工
```

### RBAC2 - 约束模型
- **互斥角色**: 用户不能同时拥有互斥的角色
- **基数约束**: 限制角色的用户数量
- **先决条件**: 需要先拥有某些角色才能获得其他角色

### RBAC3 - 统一模型
结合RBAC1和RBAC2的所有特性

## 权限命名规范

### 推荐格式
```
<action>:<resource>[:<subresource>]
```

### 示例
```
# 基础权限
read:users          # 读取用户信息
write:users         # 修改用户信息
delete:users        # 删除用户

# 细粒度权限
read:users:profile  # 读取用户资料
write:users:profile # 修改用户资料
read:posts:my       # 读取自己的文章
```

### 通配符权限
```
read:*              # 读取所有资源
*:users             # 用户的所有操作
*:*                 # 所有权限（超级管理员）
```

## 角色设计原则

### 1. 最小权限原则
角色只包含完成任务所需的最小权限集合。

### 2. 职责分离原则
将不同职责分配给不同角色，避免权限集中。

### 3. 数据抽象原则
权限应该基于业务概念，而非技术实现。

### 4. 角色层次原则
通过角色继承减少权限配置复杂度。

## 常见角色类型

### 1. 系统角色
```python
# 超级管理员
{
    "name": "super_admin",
    "description": "系统超级管理员",
    "permissions": ["*:*"]
}

# 系统管理员
{
    "name": "system_admin",
    "description": "系统管理员",
    "permissions": [
        "read:system",
        "write:system",
        "manage:users",
        "manage:roles"
    ]
}
```

### 2. 业务角色
```python
# 部门经理
{
    "name": "department_manager",
    "description": "部门经理",
    "permissions": [
        "read:users:*",
        "write:users:department",
        "read:reports:*",
        "approve:requests"
    ]
}

# 普通员工
{
    "name": "employee",
    "description": "普通员工",
    "permissions": [
        "read:users:own",
        "write:users:own",
        "read:reports:own",
        "create:requests"
    ]
}
```

### 3. 功能角色
```python
# 审计员
{
    "name": "auditor",
    "description": "审计员",
    "permissions": [
        "read:audit_logs",
        "read:system_logs",
        "export:reports"
    ]
}

# 数据分析师
{
    "name": "data_analyst",
    "description": "数据分析师",
    "permissions": [
        "read:analytics:*",
        "export:data",
        "create:reports"
    ]
}
```

## RBAC实现架构

### 1. 数据模型设计
```sql
-- 用户表
CREATE TABLE users (
    id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) UNIQUE,
    email VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP
);

-- 角色表
CREATE TABLE roles (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    description TEXT,
    created_at TIMESTAMP
);

-- 权限表
CREATE TABLE permissions (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    description TEXT,
    resource VARCHAR(100),
    action VARCHAR(100)
);

-- 用户角色关联表
CREATE TABLE user_roles (
    user_id VARCHAR(50),
    role_id VARCHAR(50),
    assigned_at TIMESTAMP,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (role_id) REFERENCES roles(id)
);

-- 角色权限关联表
CREATE TABLE role_permissions (
    role_id VARCHAR(50),
    permission_id VARCHAR(50),
    PRIMARY KEY (role_id, permission_id),
    FOREIGN KEY (role_id) REFERENCES roles(id),
    FOREIGN KEY (permission_id) REFERENCES permissions(id)
);
```

### 2. 权限检查流程
```python
def check_permission(user_id, required_permission):
    """
    检查用户是否拥有指定权限
    """
    # 1. 获取用户角色
    user_roles = get_user_roles(user_id)

    # 2. 获取角色权限
    user_permissions = set()
    for role in user_roles:
        role_permissions = get_role_permissions(role)
        user_permissions.update(role_permissions)

    # 3. 权限匹配
    return any(
        permission_matches(user_perm, required_permission)
        for user_perm in user_permissions
    )

def permission_matches(user_permission, required_permission):
    """
    检查权限是否匹配，支持通配符
    """
    if user_permission == required_permission:
        return True

    # 解析权限
    user_action, user_resource = parse_permission(user_permission)
    req_action, req_resource = parse_permission(required_permission)

    # 检查匹配
    action_match = (user_action == "*" or user_action == req_action)
    resource_match = (user_resource == "*" or user_resource == req_resource)

    return action_match and resource_match
```

## 高级特性

### 1. 动态权限
```python
class DynamicPermission:
    def __init__(self, permission_template, context_extractor):
        self.permission_template = permission_template
        self.context_extractor = context_extractor

    def evaluate(self, user, context):
        """根据上下文动态评估权限"""
        dynamic_values = self.context_extractor(user, context)
        return self.permission_template.format(**dynamic_values)

# 示例：只能查看自己部门的用户
own_department_permission = DynamicPermission(
    "read:users:department_{department_id}",
    lambda user, ctx: {"department_id": user.department_id}
)
```

### 2. 时间约束权限
```python
class TimeBasedPermission:
    def __init__(self, base_permission, time_constraints):
        self.base_permission = base_permission
        self.time_constraints = time_constraints

    def is_valid(self, current_time):
        """检查当前时间是否满足约束"""
        return all(
            constraint.is_valid(current_time)
            for constraint in self.time_constraints
        )

# 示例：工作时间权限
working_hours = TimeBasedPermission(
    "read:sensitive_data",
    [WorkingHoursConstraint(9, 18), WeekdayConstraint()]
)
```

### 3. 条件权限
```python
class ConditionalPermission:
    def __init__(self, base_permission, condition):
        self.base_permission = base_permission
        self.condition = condition

    def check(self, user, resource, context):
        """检查条件是否满足"""
        return self.condition.evaluate(user, resource, context)

# 示例：只能查看低于自己级别的数据
level_based_permission = ConditionalPermission(
    "read:reports",
    lambda user, resource, ctx: user.level >= resource.level
)
```

## 性能优化

### 1. 权限缓存
```python
from functools import lru_cache
import redis

class PermissionCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_timeout = 300  # 5分钟

    @lru_cache(maxsize=1000)
    def get_user_permissions(self, user_id):
        """获取用户权限（带缓存）"""
        cache_key = f"user_permissions:{user_id}"

        # 尝试从Redis获取
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 从数据库获取
        permissions = self._fetch_user_permissions(user_id)

        # 缓存结果
        self.redis.setex(
            cache_key,
            self.cache_timeout,
            json.dumps(permissions)
        )

        return permissions
```

### 2. 批量权限检查
```python
def batch_check_permissions(user_id, required_permissions):
    """批量检查用户权限"""
    user_permissions = get_user_permissions(user_id)

    results = {}
    for permission in required_permissions:
        results[permission] = any(
            permission_matches(user_perm, permission)
            for user_perm in user_permissions
        )

    return results
```

### 3. 权限预计算
```python
class PermissionPrecomputer:
    def __init__(self):
        self.permission_matrix = {}

    def precompute_permissions(self):
        """预计算所有角色的权限组合"""
        for role in get_all_roles():
            role_permissions = get_role_permissions(role)

            for user in get_users_with_role(role):
                if user not in self.permission_matrix:
                    self.permission_matrix[user] = set()
                self.permission_matrix[user].update(role_permissions)

    def get_user_permissions_fast(self, user_id):
        """快速获取用户权限"""
        return self.permission_matrix.get(user_id, set())
```

## 安全考虑

### 1. 权限提升攻击防护
```python
def prevent_privilege_escalation(user, target_permission):
    """防止权限提升攻击"""
    user_permissions = get_user_permissions(user.id)

    # 检查用户是否试图获得自己没有的权限
    if not any(
        permission_matches(perm, target_permission)
        for perm in user_permissions
    ):
        log_security_event("权限提升尝试", user, target_permission)
        return False

    return True
```

### 2. 角色冲突检测
```python
def detect_role_conflicts(user_roles):
    """检测角色冲突"""
    conflict_pairs = [
        ("auditor", "finance_manager"),
        ("developer", "production_admin"),
        ("tester", "release_manager")
    ]

    for role1, role2 in conflict_pairs:
        if role1 in user_roles and role2 in user_roles:
            return True, f"角色冲突: {role1} 和 {role2}"

    return False, None
```

### 3. 审计日志
```python
def log_permission_check(user_id, permission, result, context):
    """记录权限检查日志"""
    audit_log = {
        "timestamp": datetime.utcnow(),
        "user_id": user_id,
        "permission": permission,
        "result": result,
        "ip_address": context.get("ip_address"),
        "user_agent": context.get("user_agent"),
        "resource": context.get("resource")
    }

    save_audit_log(audit_log)
```

## 监控和分析

### 1. 权限使用统计
```python
class PermissionAnalytics:
    def track_permission_usage(self, user_id, permission, result):
        """跟踪权限使用情况"""
        key = f"perm_usage:{permission}"
        self.redis.incr(key)

        if result:
            success_key = f"perm_success:{permission}"
            self.redis.incr(success_key)

    def get_permission_stats(self):
        """获取权限使用统计"""
        stats = {}
        for permission in get_all_permissions():
            total = self.redis.get(f"perm_usage:{permission}") or 0
            success = self.redis.get(f"perm_success:{permission}") or 0
            stats[permission] = {
                "total": int(total),
                "success": int(success),
                "success_rate": int(success) / int(total) if int(total) > 0 else 0
            }
        return stats
```

### 2. 异常检测
```python
def detect_permission_anomalies(user_id):
    """检测权限使用异常"""
    recent_checks = get_recent_permission_checks(user_id, hours=24)

    # 检测权限失败率过高
    failure_rate = sum(1 for check in recent_checks if not check.result) / len(recent_checks)
    if failure_rate > 0.5:
        alert_security_team(f"用户 {user_id} 权限失败率过高: {failure_rate:.2%}")

    # 检测异常时间访问
    unusual_hours = [check for check in recent_checks if check.timestamp.hour < 6 or check.timestamp.hour > 22]
    if len(unusual_hours) > len(recent_checks) * 0.3:
        alert_security_team(f"用户 {user_id} 在异常时间访问系统")
```

## 实际应用案例

### 1. 电商系统RBAC
```python
# 顾客角色
customer_role = {
    "permissions": [
        "read:products",
        "create:orders",
        "read:orders:own",
        "write:profile:own"
    ]
}

# 商家角色
merchant_role = {
    "permissions": [
        "read:products:own",
        "write:products:own",
        "read:orders:own",
        "manage:inventory:own"
    ]
}

# 平台管理员
platform_admin_role = {
    "permissions": [
        "read:*",
        "write:products:*",
        "manage:users",
        "manage:merchants"
    ]
}
```

### 2. 企业内部系统RBAC
```python
# 员工角色
employee_role = {
    "permissions": [
        "read:documents:department",
        "write:documents:own",
        "create:requests",
        "read:requests:own"
    ]
}

# 部门经理
manager_role = {
    "permissions": [
        "read:documents:department",
        "write:documents:department",
        "approve:requests:department",
        "read:reports:department"
    ]
}

# HR角色
hr_role = {
    "permissions": [
        "read:employees:*",
        "write:employees:*",
        "manage:leaves",
        "manage:salaries"
    ]
}
```

## 最佳实践总结

### 1. 设计原则
- ✅ 遵循最小权限原则
- ✅ 使用角色层次简化管理
- ✅ 定期审查权限分配
- ✅ 实现权限审计机制

### 2. 实现要点
- ✅ 支持通配符权限匹配
- ✅ 实现高效的权限缓存
- ✅ 提供完整的权限检查API
- ✅ 记录详细的权限使用日志

### 3. 安全措施
- ✅ 防止权限提升攻击
- ✅ 检测角色冲突
- ✅ 监控异常权限使用
- ✅ 实现权限回收机制

## 参考资源

- [NIST RBAC标准](https://csrc.nist.gov/projects/rbac)
- [OWASP访问控制指南](https://owasp.org/www-project-access-control/)
- [RFC 6585 - RBAC Profile](https://tools.ietf.org/html/rfc6585)