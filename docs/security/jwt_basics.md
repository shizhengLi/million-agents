# JWT认证基础知识

## 什么是JWT

JWT (JSON Web Token) 是一种开放标准 (RFC 7519)，用于在各方之间安全地传输信息作为JSON对象。JWT可以被验证和信任，因为它是数字签名的。

## JWT结构

JWT由三个部分组成，用点(.)分隔：

```
xxxxx.yyyyy.zzzzz
```

### 1. Header（头部）
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

- `alg`: 签名算法 (如HS256, RS256)
- `typ`: 令牌类型，通常为JWT

### 2. Payload（载荷）
```json
{
  "user_id": "123456",
  "username": "john_doe",
  "role": "admin",
  "exp": 1516239022,
  "iat": 1516239022
}
```

**标准声明**：
- `iss`: 签发者
- `sub`: 主题
- `aud`: 接收方
- `exp`: 过期时间
- `iat`: 签发时间
- `jti`: JWT ID

**自定义声明**：
- `user_id`: 用户ID
- `username`: 用户名
- `role`: 用户角色
- `permissions`: 权限列表

### 3. Signature（签名）
```
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret_key
)
```

## JWT工作流程

```
用户登录 → 验证身份 → 生成JWT → 返回JWT → 客户端存储 → 后续请求携带JWT → 服务器验证 → 授权访问
```

## JWT的优缺点

### 优点
1. **无状态性**: 服务器不需要存储会话信息
2. **跨语言支持**: 支持多种编程语言
3. **信息丰富**: 可以在令牌中携带用户信息
4. **扩展性好**: 易于水平扩展
5. **移动端友好**: 支持移动应用和Web应用

### 缺点
1. **令牌大小**: 比Session ID更大
2. **无法撤销**: 一旦签发，在过期前无法撤销
3. **安全性**: 依赖密钥的安全
4. **令牌刷新**: 需要额外的刷新机制

## JWT最佳实践

### 1. 密钥管理
```python
# 使用强密钥
SECRET_KEY = "your-very-strong-secret-key-with-special-characters"

# 定期轮换密钥
# 在生产环境中，建议每90天轮换一次密钥
```

### 2. 过期时间设置
```python
# 访问令牌：15-30分钟
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 刷新令牌：7-30天
REFRESH_TOKEN_EXPIRE_DAYS = 7
```

### 3. 敏感信息处理
```python
# 不在JWT中存储敏感信息
# ❌ 错误示例
payload = {
    "user_id": "123",
    "password": "secret"  # 不要存储密码
}

# ✅ 正确示例
payload = {
    "user_id": "123",
    "role": "user",
    "permissions": ["read", "write"]
}
```

### 4. 算法选择
```python
# 开发环境：HS256（对称加密）
algorithm = "HS256"

# 生产环境：RS256（非对称加密）
algorithm = "RS256"
```

## 常见攻击及防护

### 1. 令牌重放攻击
**防护措施**：
- 使用短期过期的访问令牌
- 实现令牌黑名单机制
- 使用随机nonce

### 2. 令牌泄露
**防护措施**：
- 使用HTTPS传输
- 安全存储客户端令牌
- 实现令牌刷新机制

### 3. 算法混淆攻击
**防护措施**：
- 明确指定签名算法
- 不使用"none"算法
- 验证算法的一致性

## JWT vs Session

| 特性 | JWT | Session |
|------|-----|---------|
| 存储位置 | 客户端 | 服务器 |
| 扩展性 | 好 | 一般 |
| 安全性 | 中等 | 高 |
| 状态 | 无状态 | 有状态 |
| 大小 | 较大 | 较小 |
| 跨域支持 | 好 | 一般 |

## 示例代码

### 生成JWT
```python
import jwt
from datetime import datetime, timedelta

def generate_jwt(user_id, username, role):
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "iat": datetime.utcnow(),
        "type": "access"
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token
```

### 验证JWT
```python
def verify_jwt(token):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"],
            options={"require": ["exp", "iat"]}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("令牌已过期")
    except jwt.InvalidTokenError:
        raise Exception("无效令牌")
```

## 常用库

### Python
- **PyJWT**: 官方推荐的JWT库
- **Authlib**: 功能更全面的认证库

### JavaScript
- **jsonwebtoken**: Node.js JWT库
- **jsrsasign**: 客户端JWT库

### Java
- **jjwt**: Java JWT库
- **nimbus-jose-jwt**: 功能丰富的JWT库

## 相关标准

- **RFC 7519**: JWT标准
- **RFC 7515**: JWS标准
- **RFC 7516**: JWE标准
- **RFC 7517**: JWK标准

## 参考资源

- [JWT官方网站](https://jwt.io/)
- [JWT RFC 7519](https://tools.ietf.org/html/rfc7519)
- [OWASP JWT Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_Cheat_Sheet.html)