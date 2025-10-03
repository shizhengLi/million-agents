# JWT认证系统面试题集

## 基础概念题

### 1. 什么是JWT？它的主要组成部分是什么？
**答案**：
JWT (JSON Web Token) 是一种开放标准(RFC 7519)，用于在各方之间安全地传输信息。它由三部分组成：

1. **Header (头部)**: 包含令牌类型和签名算法
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

2. **Payload (载荷)**: 包含用户信息和声明
```json
{
  "user_id": "123",
  "username": "john",
  "exp": 1516239022
}
```

3. **Signature (签名)**: 用于验证令牌完整性
```
HMACSHA256(base64UrlEncode(header) + "." + base64UrlEncode(payload), secret)
```

### 2. JWT相比Session有哪些优势？
**答案**：
- **无状态性**: 服务器不需要存储会话信息
- **跨域支持**: 适合分布式系统和微服务架构
- **扩展性好**: 易于水平扩展和负载均衡
- **移动端友好**: 支持原生移动应用
- **信息丰富**: 可以在令牌中携带用户信息

### 3. JWT的常见使用场景有哪些？
**答案**：
- 用户身份认证和授权
- API访问控制
- 单点登录(SSO)
- 微服务间通信
- 信息交换(安全传输)

## 安全性问题

### 4. JWT安全方面有哪些主要风险？如何防范？
**答案**：
**主要风险**：
1. **令牌泄露**: 通过XSS、日志泄露等方式
2. **重放攻击**: 拦截并重用令牌
3. **算法混淆攻击**: 修改算法字段
4. **密钥泄露**: 签名密钥被获取

**防范措施**：
```python
# 1. 使用HTTPS传输
# 2. 设置合理的过期时间
ACCESS_TOKEN_EXPIRE = 15  # 分钟
REFRESH_TOKEN_EXPIRE = 7  # 天

# 3. 实现令牌黑名单
class TokenBlacklist:
    def revoke_token(self, jti, exp):
        ttl = exp - current_time
        redis.setex(f"blacklist:{jti}", ttl, "revoked")

# 4. 使用强密钥和安全的算法
SECRET_KEY = "very-strong-random-key"
ALGORITHM = "RS256"  # 生产环境推荐
```

### 5. 如何防止JWT重放攻击？
**答案**：
```python
# 1. 使用短期有效的访问令牌
# 2. 实现nonce机制
def generate_jwt_with_nonce(user_data):
    payload = {
        **user_data,
        "nonce": secrets.token_urlsafe(16),
        "exp": datetime.utcnow() + timedelta(minutes=15)
    }
    return jwt.encode(payload, SECRET_KEY, ALGORITHM)

# 3. 实现令牌使用记录
class TokenUsageTracker:
    def track_token_usage(self, jti, ip_address, user_agent):
        key = f"token_usage:{jti}"
        usage_data = {
            "ip_address": ip_address,
            "user_agent": user_agent,
            "first_used": datetime.utcnow()
        }
        redis.setex(key, 900, json.dumps(usage_data))  # 15分钟

    def is_token_reused(self, jti, ip_address, user_agent):
        stored = redis.get(f"token_usage:{jti}")
        if stored:
            data = json.loads(stored)
            return (data["ip_address"] != ip_address or
                   data["user_agent"] != user_agent)
        return False
```

### 6. JWT令牌被撤销怎么办？如何实现令牌撤销？
**答案**：
JWT本身不支持撤销，但可以通过以下方式实现：

```python
# 1. 令牌黑名单机制
class JWTRevocationService:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def revoke_token(self, token, reason="手动撤销"):
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                ttl = exp - int(time.time())
                if ttl > 0:
                    await self.redis.setex(
                        f"revoked:{jti}",
                        ttl,
                        json.dumps({
                            "revoked_at": datetime.utcnow().isoformat(),
                            "reason": reason
                        })
                    )
        except Exception as e:
            logger.error(f"撤销令牌失败: {e}")

    async def is_token_revoked(self, token):
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            return jti and await self.redis.exists(f"revoked:{jti}")
        except:
            return True
```

### 7. 如何安全地存储JWT令牌？
**答案**：
```javascript
// Web端存储
// ✅ 推荐：使用HttpOnly Cookie
document.cookie = `token=${jwt}; HttpOnly; Secure; SameSite=Strict`;

// ⚠️ 注意：LocalStorage不是最佳选择
// localStorage.setItem('token', jwt); // 不推荐，易受XSS攻击

// 移动端存储
// ✅ Android - 使用EncryptedSharedPreferences
SharedPreferences sharedPreferences = context.getSharedPreferences(
    "auth", Context.MODE_PRIVATE);
EncryptedSharedPreferences encryptedPrefs = EncryptedSharedPreferences.create(
    "auth",
    MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC),
    sharedPreferences,
    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
);

// ✅ iOS - 使用Keychain
let keychain = Keychain(service: "com.app.auth")
keychain["jwt"] = jwt
```

## 实现细节题

### 8. 实现一个完整的JWT认证中间件
**答案**：
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime

security = HTTPBearer()

class JWTMiddleware:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.blacklist = TokenBlacklist()

    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """验证JWT令牌"""
        try:
            token = credentials.credentials

            # 1. 验证令牌格式和签名
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "iat", "jti"]}
            )

            # 2. 检查令牌是否被撤销
            if await self.blacklist.is_token_revoked(token):
                raise HTTPException(status_code=401, detail="令牌已被撤销")

            # 3. 检查令牌类型
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="令牌类型错误")

            # 4. 检查用户状态
            user = await get_user_by_id(payload["user_id"])
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="用户不存在或已被禁用")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"无效令牌: {str(e)}")

# 使用示例
app = FastAPI()
auth_middleware = JWTMiddleware(SECRET_KEY)

@app.get("/protected")
async def protected_route(payload: dict = Depends(auth_middleware.verify_token)):
    return {"message": f"Hello, {payload['username']}!"}
```

### 9. 如何实现JWT令牌刷新机制？
**答案**：
```python
from datetime import datetime, timedelta
import secrets

class TokenRefreshService:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.access_token_expire = timedelta(minutes=15)
        self.refresh_token_expire = timedelta(days=7)

    async def refresh_tokens(self, refresh_token: str):
        """刷新令牌对"""
        try:
            # 1. 验证刷新令牌
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=["HS256"]
            )

            if payload.get("type") != "refresh":
                raise InvalidToken("无效的令牌类型")

            # 2. 获取用户信息
            user = await get_user_by_id(payload["user_id"])
            if not user or not user.is_active:
                raise InvalidToken("用户不存在或已被禁用")

            # 3. 生成新的令牌对
            new_tokens = await self._generate_token_pair(user)

            # 4. 撤销旧的刷新令牌
            await self._revoke_refresh_token(refresh_token)

            # 5. 记录刷新事件
            await self._log_refresh_event(user.user_id)

            return new_tokens

        except jwt.ExpiredSignatureError:
            raise InvalidToken("刷新令牌已过期")
        except jwt.InvalidTokenError:
            raise InvalidToken("无效的刷新令牌")

    async def _generate_token_pair(self, user):
        """生成令牌对"""
        now = datetime.utcnow()

        # 访问令牌
        access_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "exp": now + self.access_token_expire,
            "iat": now,
            "type": "access",
            "jti": secrets.token_urlsafe(16)
        }

        # 刷新令牌
        refresh_payload = {
            "user_id": user.user_id,
            "exp": now + self.refresh_token_expire,
            "iat": now,
            "type": "refresh",
            "jti": secrets.token_urlsafe(16)
        }

        access_token = jwt.encode(access_payload, self.secret_key, "HS256")
        refresh_token = jwt.encode(refresh_payload, self.secret_key, "HS256")

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": int(self.access_token_expire.total_seconds())
        }
```

### 10. 如何在微服务架构中实现JWT认证？
**答案**：
```python
# 1. 认证服务 - 负责令牌生成和验证
class AuthService:
    def __init__(self):
        self.jwt_auth = JWTAuthenticator()
        self.redis_client = redis.Redis()

    async def authenticate_and_generate_tokens(self, credentials):
        user = await self.authenticate_user(credentials)
        tokens = self.jwt_auth.generate_token_pair(user.to_dict())

        # 缓存令牌信息用于其他服务验证
        await self.cache_token_info(user.user_id, tokens)
        return tokens

    async def validate_token_for_service(self, token, service_name):
        """为其他微服务验证令牌"""
        try:
            payload = self.jwt_auth.verify_token(token)

            # 检查服务访问权限
            if not await self.check_service_permission(
                payload["user_id"], service_name
            ):
                raise PermissionError("无权访问该服务")

            return payload
        except Exception as e:
            raise InvalidToken(f"令牌验证失败: {e}")

# 2. 其他微服务 - 通过API调用验证令牌
class MicroserviceClient:
    def __init__(self, auth_service_url):
        self.auth_service_url = auth_service_url

    async def verify_token(self, token: str):
        """向认证服务验证令牌"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.auth_service_url}/validate",
                json={"token": token, "service": "user-service"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise InvalidToken("令牌验证失败")

# 3. API网关 - 统一认证入口
class APIGatewayAuth:
    def __init__(self, auth_service):
        self.auth_service = auth_service

    async def authenticate_request(self, request):
        """网关层认证"""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="缺少认证令牌")

        token = auth_header.split(" ")[1]

        # 向认证服务验证令牌
        user_info = await self.auth_service.validate_token_for_service(
            token, request.path.split("/")[1]
        )

        # 将用户信息添加到请求头
        request.headers["X-User-ID"] = user_info["user_id"]
        request.headers["X-User-Role"] = user_info["role"]

        return user_info
```

## 性能优化题

### 11. 如何优化JWT验证的性能？
**答案**：
```python
import asyncio
from functools import lru_cache
import time

class HighPerformanceJWTValidator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.redis_client = redis.Redis()
        self.cache_size = 10000

    @lru_cache(maxsize=1000)
    def _decode_jwt_cached(self, token: str):
        """内存缓存JWT解码结果"""
        return jwt.decode(token, self.secret_key, algorithms=["HS256"])

    async def validate_token_optimized(self, token: str):
        """高性能令牌验证"""
        # 1. 内存缓存检查
        try:
            cache_key = f"jwt_cache:{hash(token)}"
            cached_result = self._decode_jwt_cached(token)
        except Exception:
            pass

        # 2. Redis分布式缓存
        redis_key = f"jwt_valid:{token[:16]}"  # 使用令牌前16位作为键
        cached_payload = await self.redis_client.get(redis_key)

        if cached_payload:
            payload = json.loads(cached_payload)
            if not self._is_token_expired(payload):
                return payload

        # 3. 实际验证
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                options={"require": ["exp", "iat"]}
            )

            # 缓存验证结果（短期，避免过期令牌被缓存）
            ttl = max(1, payload["exp"] - int(time.time()))
            await self.redis_client.setex(
                redis_key,
                min(ttl, 300),  # 最多缓存5分钟
                json.dumps(payload)
            )

            return payload

        except jwt.ExpiredSignatureError:
            # 清理过期缓存
            await self.redis_client.delete(redis_key)
            raise
        except jwt.InvalidTokenError:
            raise

    def _is_token_expired(self, payload: dict) -> bool:
        """检查令牌是否过期"""
        return payload.get("exp", 0) <= time.time()

# 4. 批量验证优化
async def batch_validate_tokens(self, tokens: List[str]) -> Dict[str, dict]:
    """批量验证令牌"""
    # 使用异步并发验证
    tasks = [
        self.validate_token_optimized(token)
        for token in tokens
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        token: result if not isinstance(result, Exception) else None
        for token, result in zip(tokens, results)
    }
```

### 12. 如何设计JWT的缓存策略？
**答案**：
```python
class JWTCacheStrategy:
    def __init__(self, redis_client, memory_cache_size=5000):
        self.redis = redis_client
        self.memory_cache = TTLCache(maxsize=memory_cache_size, ttl=300)
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "redis_hits": 0
        }

    async def get_cached_payload(self, token: str) -> Optional[dict]:
        """分层缓存获取载荷"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # L1: 内存缓存
        if token_hash in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[token_hash]

        # L2: Redis缓存
        redis_key = f"jwt_payload:{token_hash}"
        cached_data = await self.redis.get(redis_key)

        if cached_data:
            self.cache_stats["redis_hits"] += 1
            payload = json.loads(cached_data)

            # 回填内存缓存
            self.memory_cache[token_hash] = payload
            return payload

        self.cache_stats["misses"] += 1
        return None

    async def cache_payload(self, token: str, payload: dict, ttl: int = None):
        """缓存载荷"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # 设置内存缓存
        self.memory_cache[token_hash] = payload

        # 设置Redis缓存
        redis_key = f"jwt_payload:{token_hash}"
        cache_ttl = min(ttl or 300, payload.get("exp", 0) - int(time.time()))

        if cache_ttl > 0:
            await self.redis.setex(
                redis_key,
                cache_ttl,
                json.dumps(payload)
            )

    async def invalidate_cache(self, token: str):
        """失效缓存"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # 清除内存缓存
        self.memory_cache.pop(token_hash, None)

        # 清除Redis缓存
        redis_key = f"jwt_payload:{token_hash}"
        await self.redis.delete(redis_key)

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        total_requests = (
            self.cache_stats["hits"] +
            self.cache_stats["misses"]
        )

        return {
            **self.cache_stats,
            "hit_rate": (
                self.cache_stats["hits"] / total_requests
                if total_requests > 0 else 0
            ),
            "redis_hit_rate": (
                self.cache_stats["redis_hits"] / total_requests
                if total_requests > 0 else 0
            )
        }
```

## 故障处理题

### 13. JWT令牌过期如何处理？
**答案**：
```javascript
// 前端自动刷新令牌
class TokenManager {
  constructor() {
    this.refreshPromise = null;
  }

  async makeRequest(url, options = {}) {
    let token = this.getAccessToken();

    try {
      return await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${token}`
        }
      });
    } catch (error) {
      if (error.status === 401) {
        // 令牌过期，尝试刷新
        const newToken = await this.refreshToken();

        // 重试原请求
        return fetch(url, {
          ...options,
          headers: {
            ...options.headers,
            'Authorization': `Bearer ${newToken}`
          }
        });
      }
      throw error;
    }
  }

  async refreshToken() {
    // 防止并发刷新
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = this.doRefreshToken();

    try {
      const result = await this.refreshPromise;
      return result.access_token;
    } finally {
      this.refreshPromise = null;
    }
  }

  async doRefreshToken() {
    const refreshToken = this.getRefreshToken();

    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        refresh_token: refreshToken
      })
    });

    if (!response.ok) {
      // 刷新失败，重定向到登录页
      this.redirectToLogin();
      throw new Error('Token refresh failed');
    }

    const data = await response.json();
    this.setTokens(data);
    return data;
  }

  redirectToLogin() {
    // 清除本地令牌
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');

    // 重定向到登录页
    window.location.href = '/login';
  }
}
```

### 14. 如何处理JWT密钥轮换？
**答案**：
```python
class JWTKeyRotation:
    def __init__(self):
        self.current_key_id = "key_2024_01"
        self.keys = {
            "key_2024_01": "current_secret_key",
            "key_2023_12": "previous_secret_key",
            "key_2023_11": "older_secret_key"
        }
        self.key_expiry = {
            "key_2024_01": datetime(2024, 2, 1),
            "key_2023_12": datetime(2024, 1, 1),
            "key_2023_11": datetime(2023, 12, 1)
        }

    def generate_token_with_key_id(self, payload: dict) -> str:
        """生成带密钥ID的令牌"""
        payload["kid"] = self.current_key_id
        payload["iat"] = datetime.utcnow()

        return jwt.encode(
            payload,
            self.keys[self.current_key_id],
            algorithm="HS256"
        )

    def verify_token_with_key_rotation(self, token: str) -> dict:
        """支持密钥轮换的令牌验证"""
        try:
            # 无密钥信息，尝试所有密钥
            unverified_header = jwt.get_unverified_header(token)
            key_id = unverified_header.get("kid")

            if key_id and key_id in self.keys:
                # 使用指定密钥验证
                return jwt.decode(
                    token,
                    self.keys[key_id],
                    algorithms=["HS256"],
                    options={"verify_exp": True}
                )
            else:
                # 尝试所有密钥
                for key_id, secret_key in self.keys.items():
                    try:
                        payload = jwt.decode(
                            token,
                            secret_key,
                            algorithms=["HS256"],
                            options={"verify_exp": True}
                        )
                        # 验证成功，记录密钥使用情况
                        self.log_key_usage(key_id)
                        return payload
                    except jwt.InvalidTokenError:
                        continue

                raise jwt.InvalidTokenError("无法使用任何密钥验证令牌")

        except Exception as e:
            raise jwt.InvalidTokenError(f"令牌验证失败: {e}")

    async def rotate_keys(self):
        """定期轮换密钥"""
        new_key_id = f"key_{datetime.now().strftime('%Y_%m')}"
        new_secret_key = secrets.token_urlsafe(64)

        # 添加新密钥
        self.keys[new_key_id] = new_secret_key
        self.key_expiry[new_key_id] = datetime.utcnow() + timedelta(days=30)

        # 更新当前密钥
        self.current_key_id = new_key_id

        # 清理过期密钥
        await self.cleanup_expired_keys()

        # 记录密钥轮换事件
        await self.log_key_rotation(new_key_id)

    async def cleanup_expired_keys(self):
        """清理过期密钥"""
        now = datetime.utcnow()
        expired_keys = [
            key_id for key_id, expiry_time in self.key_expiry.items()
            if expiry_time < now
        ]

        for key_id in expired_keys:
            del self.keys[key_id]
            del self.key_expiry[key_id]
            logger.info(f"清理过期密钥: {key_id}")
```

## 场景设计题

### 15. 设计一个支持SSO的JWT认证系统
**答案**：
```python
class SSOJWTProvider:
    def __init__(self, issuer_url: str, client_apps: Dict[str, dict]):
        self.issuer_url = issuer_url
        self.client_apps = client_apps  # {app_id: {secret, redirect_uris}}
        self.jwt_auth = JWTAuthenticator()
        self.session_store = RedisStore()

    async def initiate_sso_login(self, client_id: str, redirect_uri: str, state: str):
        """发起SSO登录"""
        # 1. 验证客户端
        if client_id not in self.client_apps:
            raise InvalidClient("无效的客户端ID")

        client_config = self.client_apps[client_id]
        if redirect_uri not in client_config["redirect_uris"]:
            raise InvalidRedirectURI("无效的重定向URI")

        # 2. 生成授权码
        auth_code = secrets.token_urlsafe(32)
        await self.session_store.set(
            f"auth_code:{auth_code}",
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "state": state,
                "expires_at": datetime.utcnow() + timedelta(minutes=10)
            },
            ttl=600
        )

        # 3. 重定向到登录页面
        login_url = f"{self.issuer_url}/login?auth_code={auth_code}"
        return RedirectResponse(url=login_url)

    async def authenticate_and_authorize(self, auth_code: str, credentials: LoginCredentials):
        """认证并授权"""
        # 1. 验证授权码
        auth_data = await self.session_store.get(f"auth_code:{auth_code}")
        if not auth_data:
            raise InvalidAuthCode("无效的授权码")

        # 2. 用户认证
        user = await self.authenticate_user(credentials)
        if not user:
            raise AuthenticationError("认证失败")

        # 3. 生成授权响应
        client_config = self.client_apps[auth_data["client_id"]]

        # 生成ID Token (OpenID Connect)
        id_token = self.generate_id_token(user, auth_data["client_id"])

        # 生成Access Token
        access_token = self.jwt_auth.generate_access_token({
            "user_id": user.user_id,
            "username": user.username,
            "aud": auth_data["client_id"],
            "iss": self.issuer_url
        })

        # 构建重定向URL
        redirect_params = {
            "access_token": access_token,
            "id_token": id_token,
            "token_type": "Bearer",
            "state": auth_data["state"]
        }

        redirect_url = f"{auth_data['redirect_uri']}?{urlencode(redirect_params)}"
        return RedirectResponse(url=redirect_url)

    def generate_id_token(self, user: User, client_id: str) -> str:
        """生成ID Token"""
        now = datetime.utcnow()
        payload = {
            "iss": self.issuer_url,
            "sub": user.user_id,
            "aud": client_id,
            "exp": now + timedelta(hours=1),
            "iat": now,
            "email": user.email,
            "name": user.full_name,
            "preferred_username": user.username
        }

        # 使用客户端密钥签名
        client_secret = self.client_apps[client_id]["secret"]
        return jwt.encode(payload, client_secret, algorithm="HS256")

    async def validate_client_token(self, token: str, client_id: str):
        """客户端验证令牌"""
        try:
            client_secret = self.client_apps[client_id]["secret"]
            payload = jwt.decode(
                token,
                client_secret,
                algorithms=["HS256"],
                audience=client_id,
                issuer=self.issuer_url
            )
            return payload
        except jwt.InvalidTokenError as e:
            raise InvalidToken(f"客户端令牌验证失败: {e}")
```

### 16. 如何实现JWT在移动端和Web端的统一认证？
**答案**：
```python
class UnifiedAuthSystem:
    def __init__(self):
        self.device_registry = DeviceRegistry()
        self.jwt_auth = JWTAuthenticator()

    async def authenticate_device(self, device_info: DeviceInfo, credentials: LoginCredentials):
        """设备认证"""
        # 1. 用户认证
        user = await self.authenticate_user(credentials)

        # 2. 设备注册/验证
        device = await self.device_registry.register_or_update_device(
            user.user_id,
            device_info
        )

        # 3. 生成设备特定的令牌
        device_claims = {
            "user_id": user.user_id,
            "device_id": device.device_id,
            "device_type": device.device_type,
            "trusted": device.is_trusted
        }

        access_token = self.jwt_auth.generate_access_token({
            **user.to_dict(),
            **device_claims
        })

        refresh_token = self.jwt_auth.generate_refresh_token({
            "user_id": user.user_id,
            "device_id": device.device_id
        })

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "device_id": device.device_id,
            "requires_2fa": await self.requires_2fa(user.user_id, device)
        }

    async def validate_device_token(self, token: str, device_context: dict):
        """验证设备令牌"""
        payload = self.jwt_auth.verify_token(token)

        # 1. 验证设备信息
        device_id = payload.get("device_id")
        if not device_id:
            raise InvalidToken("缺少设备信息")

        device = await self.device_registry.get_device(device_id)
        if not device or device.user_id != payload["user_id"]:
            raise InvalidToken("设备验证失败")

        # 2. 检查设备状态
        if device.is_blocked:
            raise DeviceBlocked("设备已被阻止")

        # 3. 检查信任设备
        if not device.is_trusted:
            # 需要额外验证
            if not await self.verify_device_trust(device, device_context):
                raise DeviceNotTrusted("设备未受信任")

        return payload

class DeviceRegistry:
    def __init__(self):
        self.redis_client = redis.Redis()

    async def register_or_update_device(self, user_id: str, device_info: DeviceInfo) -> Device:
        """注册或更新设备"""
        device_id = self.generate_device_id(device_info)

        existing_device = await self.get_device(device_id)

        if existing_device:
            # 更新设备信息
            await self.update_device_info(device_id, device_info)
            return existing_device
        else:
            # 注册新设备
            device = Device(
                device_id=device_id,
                user_id=user_id,
                device_type=device_info.device_type,
                device_name=device_info.device_name,
                is_trusted=False,  # 新设备默认不受信任
                created_at=datetime.utcnow()
            )

            await self.save_device(device)
            await self.notify_new_device(device)

            return device

    def generate_device_id(self, device_info: DeviceInfo) -> str:
        """生成设备ID"""
        # 基于设备指纹生成唯一ID
        fingerprint_data = f"{device_info.device_type}_{device_info.os}_{device_info.browser}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
```

## 高级概念题

### 17. 什么是OpenID Connect？它和JWT的关系是什么？
**答案**：
OpenID Connect (OIDC) 是在OAuth 2.0协议之上的身份层，它使用JWT作为ID Token格式。

**主要组件**：
1. **ID Token**: JWT格式，包含用户身份信息
2. **UserInfo Endpoint**: 获取用户详细信息的端点
3. **Discovery**: 服务发现机制

```python
class OpenIDConnectProvider:
    def __init__(self, issuer_url: str):
        self.issuer_url = issuer_url
        self.jwt_auth = JWTAuthenticator()

    def generate_id_token(self, user: User, client_id: str, nonce: str = None) -> str:
        """生成OpenID Connect ID Token"""
        now = datetime.utcnow()

        payload = {
            # 标准声明
            "iss": self.issuer_url,  # 发行者
            "sub": user.user_id,     # 主题（用户ID）
            "aud": client_id,        # 受众（客户端ID）
            "exp": now + timedelta(hours=1),  # 过期时间
            "iat": now,              # 签发时间
            "auth_time": user.last_login_at,  # 认证时间

            # 可选声明
            "nonce": nonce,
            "acr": "urn:mace:incommon:iap:bronze",  # 认证上下文

            # 用户信息
            "email": user.email,
            "email_verified": user.email_verified,
            "name": user.full_name,
            "preferred_username": user.username,
            "picture": user.avatar_url
        }

        return self.jwt_auth.generate_access_token(payload)

    async def discover_configuration(self) -> dict:
        """OpenID Connect发现"""
        return {
            "issuer": self.issuer_url,
            "authorization_endpoint": f"{self.issuer_url}/auth",
            "token_endpoint": f"{self.issuer_url}/token",
            "userinfo_endpoint": f"{self.issuer_url}/userinfo",
            "jwks_uri": f"{self.issuer_url}/.well-known/jwks.json",
            "scopes_supported": ["openid", "profile", "email"],
            "response_types_supported": ["code", "id_token", "token id_token"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"]
        }
```

### 18. 如何实现JWT的JWK (JSON Web Key) 密钥管理？
**答案**：
```python
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class JWKManager:
    def __init__(self):
        self.keys = {}
        self.key_store_period = timedelta(hours=24)
        self.rotation_period = timedelta(days=30)

    async def generate_key_pair(self, key_id: str, use: str = "sig") -> dict:
        """生成密钥对"""
        # 生成RSA密钥对
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        public_key = private_key.public_key()

        # 序列化私钥
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        # 序列化公钥
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # 提取公钥参数（用于JWK）
        public_numbers = public_key.public_numbers()

        jwk = {
            "kty": "RSA",
            "kid": key_id,
            "use": use,
            "alg": "RS256",
            "n": self.base64url_encode(public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, 'big')),
            "e": self.base64url_encode(public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, 'big')),
            "x5t": self.generate_thumbprint(public_pem),
            "x5c": [self.base64_encode(public_pem)]
        }

        # 存储密钥
        self.keys[key_id] = {
            "jwk": jwk,
            "private_key": private_pem.decode(),
            "public_key": public_pem.decode(),
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + self.rotation_period
        }

        return jwk

    def get_jwks(self) -> dict:
        """获取JWKS (JSON Web Key Set)"""
        active_keys = []

        for key_id, key_data in self.keys.items():
            if datetime.utcnow() < key_data["expires_at"]:
                active_keys.append(key_data["jwk"])

        return {"keys": active_keys}

    def sign_jwt(self, payload: dict, key_id: str) -> str:
        """使用指定密钥签名JWT"""
        if key_id not in self.keys:
            raise KeyError(f"密钥 {key_id} 不存在")

        key_data = self.keys[key_id]
        private_key = serialization.load_pem_private_key(
            key_data["private_key"].encode(),
            password=None,
            backend=default_backend()
        )

        # 添加密钥ID到头部
        header = {"kid": key_id, "alg": "RS256", "typ": "JWT"}

        return jwt.encode(
            payload,
            private_key,
            algorithm="RS256",
            headers=header
        )

    def verify_jwt(self, token: str) -> dict:
        """验证JWT"""
        try:
            # 获取令牌头部信息
            unverified_header = jwt.get_unverified_header(token)
            key_id = unverified_header.get("kid")

            if not key_id or key_id not in self.keys:
                raise jwt.InvalidTokenError("无效的密钥ID")

            key_data = self.keys[key_id]
            public_key = serialization.load_pem_public_key(
                key_data["public_key"].encode(),
                backend=default_backend()
            )

            # 验证令牌
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                options={"require": ["exp", "iat", "kid"]}
            )

            return payload

        except Exception as e:
            raise jwt.InvalidTokenError(f"令牌验证失败: {e}")

    async def rotate_keys(self):
        """密钥轮换"""
        # 生成新密钥
        new_key_id = f"key_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        await self.generate_key_pair(new_key_id)

        # 清理过期密钥
        await self.cleanup_expired_keys()

        logger.info(f"密钥轮换完成，新密钥ID: {new_key_id}")

    @staticmethod
    def base64url_encode(data: bytes) -> str:
        """Base64URL编码"""
        return base64.urlsafe_b64encode(data).decode().rstrip('=')

    @staticmethod
    def base64_encode(data: bytes) -> str:
        """Base64编码"""
        return base64.b64encode(data).decode()
```

这些面试题涵盖了JWT认证系统的方方面面，从基础概念到高级实现，从安全性考虑到性能优化，能够全面评估候选人的JWT相关技术能力。