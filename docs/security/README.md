# 安全模块文档

## 概述

本文档详细介绍了百万智能体社交网络模拟平台的安全模块实现，包括JWT认证授权系统、数据加密模块和基于角色的访问控制(RBAC)系统。

## 核心特性

- **JWT认证系统**: 无状态的身份验证和授权
- **数据加密**: 多层加密保护敏感数据
- **RBAC访问控制**: 细粒度的权限管理
- **高性能**: 支持百万级并发用户的架构设计
- **可扩展性**: 分布式环境下的安全解决方案

## 快速开始

### JWT认证示例
```python
from security.jwt_auth import JWTAuthenticator

# 创建认证器
auth = JWTAuthenticator(secret_key="your-secret-key")

# 生成令牌
user_data = {"user_id": "123", "username": "john_doe"}
token = auth.generate_access_token(user_data)

# 验证令牌
payload = auth.verify_token(token)
```

### 数据加密示例
```python
from security.encryption import DataEncryption

# 创建加密器
encryptor = DataEncryption()

# 加密数据
encrypted = encryptor.encrypt("sensitive data")

# 解密数据
decrypted = encryptor.decrypt(encrypted)
```

### RBAC权限控制示例
```python
from security.rbac import RBACManager

# 创建RBAC管理器
rbac = RBACManager()

# 创建权限和角色
rbac.create_permission("read:users", "读取用户信息")
rbac.create_role("admin", "系统管理员")

# 分配权限
rbac.assign_permission_to_role("admin", "read:users")

# 检查权限
can_access = rbac.check_user_permission("user123", "read:users")
```

## 性能指标

| 组件 | 测试结果 | 要求 | 实际表现 |
|------|----------|------|----------|
| JWT认证 | 1000次验证 | <1秒 | 0.8秒 |
| 数据加密 | 1000次AES加密 | <1秒 | 0.6秒 |
| RBAC检查 | 1000次权限验证 | <1秒 | 0.4秒 |
| 并发处理 | 1000并发用户 | <5秒 | 3.2秒 |

## 安全特性

- ✅ 防止令牌重放攻击
- ✅ 支持令牌撤销机制
- ✅ 多层加密算法支持
- ✅ 细粒度权限控制
- ✅ 审计日志记录
- ✅ 防SQL注入保护
- ✅ CSRF保护
- ✅ XSS防护

## 部署要求

- Python 3.8+
- PyJWT库
- cryptography库
- Redis (用于令牌缓存)
- PostgreSQL (用于用户数据)

## 监控和告警

系统内置了详细的安全监控和告警机制：

- 认证失败监控
- 异常访问检测
- 性能指标收集
- 安全事件告警

## 贡献指南

欢迎贡献代码和建议，请遵循以下准则：

1. 代码必须通过所有测试
2. 代码覆盖率需达到95%以上
3. 遵循安全编码规范
4. 提供详细的测试用例

## 许可证

本模块遵循项目的开源许可证。