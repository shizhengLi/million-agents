# 百万级用户认证架构

## 概述

设计支持百万级并发用户的认证系统需要考虑高性能、高可用性、安全性和扩展性。本文档详细介绍了在百万智能体社交网络项目中实现的大规模认证架构。

## 架构设计原则

### 1. 水平扩展性
- 无状态设计
- 微服务架构
- 数据库分片
- 缓存层优化

### 2. 高可用性
- 多区域部署
- 故障转移机制
- 健康检查
- 自动恢复

### 3. 安全性
- 多层安全防护
- 令牌管理
- 异常检测
- 审计日志

### 4. 性能优化
- 缓存策略
- 连接池管理
- 异步处理
- 负载均衡

## 系统架构图

```mermaid
graph TB
    subgraph "客户端层"
        A[Web客户端]
        B[移动客户端]
        C[API客户端]
    end

    subgraph "负载均衡层"
        D[Nginx/HAProxy]
    end

    subgraph "认证服务层"
        E[认证服务1]
        F[认证服务2]
        G[认证服务N]
    end

    subgraph "缓存层"
        H[Redis集群]
        I[Memcached集群]
    end

    subgraph "数据库层"
        J[用户数据库分片1]
        K[用户数据库分片2]
        L[用户数据库分片N]
    end

    subgraph "监控层"
        M[监控系统]
        N[日志系统]
        O[告警系统]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    D --> F
    D --> G
    E --> H
    F --> H
    G --> H
    E --> I
    F --> I
    G --> I
    E --> J
    F --> K
    G --> L
    E --> M
    F --> M
    G --> M
    E --> N
    F --> N
    G --> N
```

## 核心组件设计

### 1. 认证服务 (Authentication Service)

#### 服务架构
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
import asyncio
import redis.asyncio as redis
import aioredis
from datetime import datetime, timedelta

class AuthenticationService:
    def __init__(self):
        self.app = FastAPI(title="百万用户认证服务")
        self.security = HTTPBearer()
        self.redis_client = None
        self.jwt_authenticator = JWTAuthenticator()

    async def initialize(self):
        """初始化服务"""
        self.redis_client = redis.Redis.from_url(
            "redis://redis-cluster:6379",
            max_connections=1000,
            retry_on_timeout=True
        )

    @app.post("/auth/login")
    async def login(self, credentials: LoginRequest):
        """用户登录接口"""
        try:
            # 1. 验证用户凭据
            user = await self.authenticate_user(credentials)

            # 2. 生成令牌对
            tokens = await self.generate_tokens(user)

            # 3. 缓存令牌信息
            await self.cache_token_info(user.user_id, tokens)

            # 4. 记录登录日志
            await self.log_login_event(user, credentials.ip_address)

            return TokenResponse(
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_in=tokens["expires_in"],
                user_info=user.to_dict()
            )

        except AuthenticationError as e:
            await self.handle_failed_login(credentials, str(e))
            raise HTTPException(status_code=401, detail=str(e))

    async def authenticate_user(self, credentials: LoginRequest):
        """认证用户"""
        # 1. 检查是否被锁定
        if await self.is_account_locked(credentials.username):
            raise AuthenticationError("账户已锁定，请稍后再试")

        # 2. 验证密码
        user = await self.get_user_by_username(credentials.username)
        if not user or not self.verify_password(credentials.password, user.password_hash):
            await self.record_failed_attempt(credentials.username)
            raise AuthenticationError("用户名或密码错误")

        # 3. 检查账户状态
        if not user.is_active:
            raise AuthenticationError("账户已被禁用")

        # 4. 清除失败尝试记录
        await self.clear_failed_attempts(credentials.username)

        return user

    async def generate_tokens(self, user: User):
        """生成令牌对"""
        user_data = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "permissions": await self.get_user_permissions(user.user_id)
        }

        access_token = self.jwt_authenticator.generate_access_token(user_data)
        refresh_token = self.jwt_authenticator.generate_refresh_token({
            "user_id": user.user_id,
            "username": user.username
        })

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": self.jwt_authenticator.access_token_expire_minutes * 60
        }

    async def cache_token_info(self, user_id: str, tokens: dict):
        """缓存令牌信息"""
        cache_key = f"token_info:{user_id}"
        token_info = {
            "user_id": user_id,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() +
                         timedelta(minutes=self.jwt_authenticator.access_token_expire_minutes)).isoformat()
        }

        await self.redis_client.setex(
            cache_key,
            self.jwt_authenticator.access_token_expire_minutes * 60 + 300,
            json.dumps(token_info)
        )

    async def verify_token(self, token: str):
        """验证令牌"""
        try:
            # 1. JWT验证
            payload = self.jwt_authenticator.verify_token(token)

            # 2. 检查令牌是否被撤销
            if await self.is_token_revoked(token):
                raise InvalidToken("令牌已被撤销")

            # 3. 检查用户状态
            user = await self.get_user_by_id(payload["user_id"])
            if not user or not user.is_active:
                raise InvalidToken("用户不存在或已被禁用")

            # 4. 更新最后访问时间
            await self.update_last_access(payload["user_id"])

            return payload

        except (TokenExpired, InvalidToken) as e:
            await self.handle_invalid_token(token, str(e))
            raise
```

### 2. 分布式令牌管理

#### 令牌黑名单
```python
class DistributedTokenBlacklist:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.blacklist_prefix = "blacklist_token:"

    async def revoke_token(self, token: str, reason: str = None):
        """撤销令牌"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                # 设置黑名单，直到令牌过期
                ttl = exp - int(time.time())
                if ttl > 0:
                    blacklist_key = f"{self.blacklist_prefix}{jti}"
                    blacklist_data = {
                        "revoked_at": datetime.utcnow().isoformat(),
                        "reason": reason or "手动撤销"
                    }

                    await self.redis.setex(
                        blacklist_key,
                        ttl,
                        json.dumps(blacklist_data)
                    )

        except Exception as e:
            logger.error(f"撤销令牌失败: {e}")

    async def is_token_revoked(self, token: str) -> bool:
        """检查令牌是否被撤销"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")

            if jti:
                blacklist_key = f"{self.blacklist_prefix}{jti}"
                revoked_data = await self.redis.get(blacklist_key)
                return revoked_data is not None

        except Exception:
            pass

        return False
```

#### 令牌刷新机制
```python
class TokenRefreshManager:
    def __init__(self, redis_client, jwt_authenticator):
        self.redis = redis_client
        self.jwt_auth = jwt_authenticator
        self.refresh_prefix = "refresh_token:"

    async def refresh_access_token(self, refresh_token: str, ip_address: str):
        """刷新访问令牌"""
        try:
            # 1. 验证刷新令牌
            payload = self.jwt_auth.verify_token(refresh_token)

            # 2. 检查刷新令牌是否有效
            if not await self.is_refresh_token_valid(refresh_token, payload["user_id"]):
                raise AuthenticationError("刷新令牌无效或已过期")

            # 3. 获取用户信息
            user = await self.get_user_by_id(payload["user_id"])
            if not user or not user.is_active:
                raise AuthenticationError("用户不存在或已被禁用")

            # 4. 生成新的令牌对
            new_tokens = await self.generate_tokens(user)

            # 5. 撤销旧的刷新令牌
            await self.revoke_refresh_token(refresh_token)

            # 6. 缓存新的令牌信息
            await self.cache_token_info(user.user_id, new_tokens)

            # 7. 记录刷新事件
            await self.log_token_refresh(user.user_id, ip_address)

            return new_tokens

        except Exception as e:
            await self.handle_refresh_failure(refresh_token, str(e), ip_address)
            raise AuthenticationError(f"令牌刷新失败: {str(e)}")

    async def is_refresh_token_valid(self, refresh_token: str, user_id: str) -> bool:
        """检查刷新令牌是否有效"""
        refresh_key = f"{self.refresh_prefix}{user_id}"
        stored_token = await self.redis.get(refresh_key)

        return stored_token == refresh_token
```

### 3. 数据库分片策略

#### 用户数据分片
```python
class UserDatabaseSharding:
    def __init__(self):
        self.shard_count = 16  # 分片数量
        self.shards = {}
        self.initialize_shards()

    def initialize_shards(self):
        """初始化数据库分片"""
        for i in range(self.shard_count):
            shard_config = {
                'host': f'user-db-{i}',
                'port': 5432,
                'database': f'user_db_{i}',
                'user': 'app_user',
                'password': 'secure_password',
                'minsize': 10,
                'maxsize': 100
            }
            self.shards[i] = create_async_pool(shard_config)

    def get_shard_number(self, user_id: str) -> int:
        """根据用户ID计算分片编号"""
        return hash(user_id) % self.shard_count

    async def get_user_by_id(self, user_id: str):
        """根据用户ID获取用户信息"""
        shard_num = self.get_shard_number(user_id)
        pool = self.shards[shard_num]

        async with pool.acquire() as conn:
            query = """
            SELECT id, username, email, password_hash, role, is_active,
                   created_at, last_login_at
            FROM users
            WHERE id = $1
            """
            row = await conn.fetchrow(query, user_id)
            return User.from_row(row) if row else None

    async def create_user(self, user: User):
        """创建用户"""
        shard_num = self.get_shard_number(user.user_id)
        pool = self.shards[shard_num]

        async with pool.acquire() as conn:
            query = """
            INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            await conn.execute(
                query,
                user.user_id,
                user.username,
                user.email,
                user.password_hash,
                user.role,
                user.is_active,
                datetime.utcnow()
            )
```

### 4. 缓存策略

#### 多层缓存架构
```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        self.l2_cache = redis.Redis.from_url("redis://cache-cluster:6379")
        self.l3_cache = redis.Redis.from_url("redis://shared-cache:6379")

    async def get_user_info(self, user_id: str):
        """获取用户信息（多级缓存）"""
        # L1缓存：本地内存
        if user_id in self.l1_cache:
            cache_data = self.l1_cache[user_id]
            if not self.is_cache_expired(cache_data):
                return cache_data["data"]

        # L2缓存：Redis集群
        l2_key = f"user:{user_id}"
        l2_data = await self.l2_cache.get(l2_key)
        if l2_data:
            user_data = json.loads(l2_data)
            # 更新L1缓存
            self.l1_cache[user_id] = {
                "data": user_data,
                "timestamp": time.time()
            }
            return user_data

        # L3缓存：共享Redis
        l3_key = f"shared_user:{user_id}"
        l3_data = await self.l3_cache.get(l3_key)
        if l3_data:
            user_data = json.loads(l3_data)
            # 更新L1和L2缓存
            self.l1_cache[user_id] = {
                "data": user_data,
                "timestamp": time.time()
            }
            await self.l2_cache.setex(l2_key, 300, l3_data)
            return user_data

        # 缓存未命中，从数据库获取
        user_data = await self.get_user_from_db(user_id)
        if user_data:
            # 更新所有缓存层
            cache_data = json.dumps(user_data.to_dict())

            self.l1_cache[user_id] = {
                "data": user_data.to_dict(),
                "timestamp": time.time()
            }
            await self.l2_cache.setex(l2_key, 300, cache_data)
            await self.l3_cache.setex(l3_key, 1800, cache_data)

        return user_data.to_dict() if user_data else None

    def is_cache_expired(self, cache_data: dict, ttl: int = 60) -> bool:
        """检查缓存是否过期"""
        return time.time() - cache_data["timestamp"] > ttl
```

### 5. 负载均衡

#### 认证服务负载均衡
```python
class AuthenticationLoadBalancer:
    def __init__(self):
        self.services = []
        self.current_index = 0
        self.health_checker = HealthChecker()

    async def add_service(self, service_url: str):
        """添加认证服务"""
        self.services.append({
            "url": service_url,
            "healthy": True,
            "last_check": datetime.utcnow(),
            "request_count": 0,
            "response_time": 0
        })

    async def get_next_service(self) -> str:
        """获取下一个可用服务（轮询+健康检查）"""
        healthy_services = [
            service for service in self.services
            if service["healthy"]
        ]

        if not healthy_services:
            raise ServiceUnavailableException("没有可用的认证服务")

        # 选择响应时间最快的服务
        best_service = min(
            healthy_services,
            key=lambda s: s["response_time"]
        )

        best_service["request_count"] += 1
        return best_service["url"]

    async def authenticate_with_load_balancing(self, credentials: LoginRequest):
        """带负载均衡的认证"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                service_url = await self.get_next_service()

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{service_url}/auth/login",
                        json=credentials.dict(),
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            await self.update_service_stats(service_url, True)
                            return result
                        else:
                            raise AuthenticationError(f"认证失败: {response.status}")

            except Exception as e:
                await self.handle_service_failure(service_url, str(e))
                if attempt == max_retries - 1:
                    raise ServiceUnavailableException("认证服务暂时不可用")

                await asyncio.sleep(0.1 * (attempt + 1))  # 指数退避
```

### 6. 性能监控

#### 认证性能指标
```python
class AuthenticationMetrics:
    def __init__(self):
        self.prometheus_client = PrometheusClient()
        self.setup_metrics()

    def setup_metrics(self):
        """设置监控指标"""
        self.login_requests_total = self.prometheus_client.Counter(
            'auth_login_requests_total',
            'Total login requests',
            ['status', 'method']
        )

        self.login_duration = self.prometheus_client.Histogram(
            'auth_login_duration_seconds',
            'Login request duration',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )

        self.active_sessions = self.prometheus_client.Gauge(
            'auth_active_sessions',
            'Number of active sessions'
        )

        self.token_cache_hit_rate = self.prometheus_client.Gauge(
            'auth_token_cache_hit_rate',
            'Token cache hit rate'
        )

    async def record_login_attempt(self, status: str, method: str, duration: float):
        """记录登录尝试"""
        self.login_requests_total.labels(status=status, method=method).inc()
        self.login_duration.observe(duration)

    async def update_active_sessions(self, count: int):
        """更新活跃会话数"""
        self.active_sessions.set(count)

    async def update_cache_hit_rate(self, hit_rate: float):
        """更新缓存命中率"""
        self.token_cache_hit_rate.set(hit_rate)
```

### 7. 安全防护

#### 速率限制
```python
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_login_rate_limit(self, ip_address: str, username: str) -> bool:
        """检查登录速率限制"""
        # IP级别限制：每分钟最多10次尝试
        ip_key = f"rate_limit:login:ip:{ip_address}"
        ip_count = await self.redis.incr(ip_key)
        if ip_count == 1:
            await self.redis.expire(ip_key, 60)

        if ip_count > 10:
            return False

        # 用户级别限制：每分钟最多5次尝试
        user_key = f"rate_limit:login:user:{username}"
        user_count = await self.redis.incr(user_key)
        if user_count == 1:
            await self.redis.expire(user_key, 60)

        return user_count <= 5

    async def check_api_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """检查API速率限制"""
        # 根据用户角色设置不同的限制
        user_role = await self.get_user_role(user_id)

        limits = {
            "free": 100,      # 免费用户每小时100次请求
            "premium": 1000,  # 付费用户每小时1000次请求
            "vip": 10000      # VIP用户每小时10000次请求
        }

        hourly_limit = limits.get(user_role, 100)
        rate_key = f"rate_limit:api:{user_id}:{endpoint}"

        current_count = await self.redis.incr(rate_key)
        if current_count == 1:
            await self.redis.expire(rate_key, 3600)

        return current_count <= hourly_limit
```

#### 异常检测
```python
class AnomalyDetector:
    def __init__(self):
        self.redis = redis.Redis.from_url("redis://anomaly-db:6379")
        self.alert_thresholds = {
            "failed_login_rate": 0.3,      # 30%登录失败率
            "unusual_location_attempts": 5, # 异常地点登录尝试
            "concurrent_sessions": 10       # 并发会话数
        }

    async def detect_login_anomalies(self, user_id: str, ip_address: str, user_agent: str):
        """检测登录异常"""
        anomalies = []

        # 1. 检测异常登录地点
        if await self.is_unusual_location(user_id, ip_address):
            anomalies.append("异常登录地点")

        # 2. 检测异常设备
        if await self.is_unusual_device(user_id, user_agent):
            anomalies.append("异常登录设备")

        # 3. 检测短时间内多次登录
        if await self.has_rapid_logins(user_id):
            anomalies.append("频繁登录尝试")

        # 4. 检测可疑时间登录
        if await self.is_suspicious_time(user_id):
            anomalies.append("可疑时间登录")

        if anomalies:
            await self.handle_security_anomaly(user_id, ip_address, anomalies)

    async def is_unusual_location(self, user_id: str, ip_address: str) -> bool:
        """检查是否为异常登录地点"""
        # 获取用户历史登录地点
        location_key = f"user_locations:{user_id}"
        locations = await self.redis.smembers(location_key)

        if not locations:
            # 首次登录，记录地点
            await self.redis.sadd(location_key, ip_address)
            await self.redis.expire(location_key, 86400 * 30)  # 30天
            return False

        if ip_address not in locations:
            # 检查IP地理位置
            current_geo = await self.get_ip_geolocation(ip_address)

            for known_ip in locations:
                known_geo = await self.get_ip_geolocation(known_ip)
                if self.calculate_distance(current_geo, known_geo) > 1000:  # 1000km
                    return True

        return False
```

## 部署架构

### 1. Docker容器化部署

#### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.auth_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  auth-service-1:
    build: .
    environment:
      - SERVICE_ID=auth-1
      - REDIS_URL=redis://redis-cluster:6379
      - DB_SHARD_1=postgresql://user:pass@user-db-1:5432/userdb
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  auth-service-2:
    build: .
    environment:
      - SERVICE_ID=auth-2
      - REDIS_URL=redis://redis-cluster:6379
      - DB_SHARD_2=postgresql://user:pass@user-db-2:5432/userdb
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G

  redis-cluster:
    image: redis:7-alpine
    deploy:
      replicas: 6
      command: redis-server --cluster-enabled yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - auth-service-1
      - auth-service-2
```

### 2. Kubernetes部署

#### Deployment配置
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
  labels:
    app: auth-service
spec:
  replicas: 10
  selector:
    matchLabels:
      app: auth-service
  template:
    metadata:
      labels:
        app: auth-service
    spec:
      containers:
      - name: auth-service
        image: million-agents/auth-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: auth-service
spec:
  selector:
    app: auth-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 性能优化策略

### 1. 数据库优化

#### 连接池配置
```python
DATABASE_CONFIG = {
    "minsize": 20,           # 最小连接数
    "maxsize": 100,          # 最大连接数
    "command_timeout": 60,   # 命令超时时间
    "server_settings": {
        "application_name": "auth_service",
        "jit": "off",        # 关闭JIT编译
        "max_parallel_workers": 4
    }
}
```

#### 查询优化
```sql
-- 创建复合索引
CREATE INDEX idx_users_username_active ON users(username, is_active) WHERE is_active = true;
CREATE INDEX idx_users_email_active ON users(email, is_active) WHERE is_active = true;

-- 分区表设计（按时间分区）
CREATE TABLE user_login_logs (
    id BIGSERIAL,
    user_id VARCHAR(50),
    login_time TIMESTAMP,
    ip_address INET,
    user_agent TEXT
) PARTITION BY RANGE (login_time);

-- 创建月度分区
CREATE TABLE user_login_logs_2024_01 PARTITION OF user_login_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 2. 缓存优化

#### Redis集群配置
```python
REDIS_CLUSTER_CONFIG = {
    "startup_nodes": [
        {"host": "redis-node-1", "port": "7000"},
        {"host": "redis-node-2", "port": "7000"},
        {"host": "redis-node-3", "port": "7000"},
        {"host": "redis-node-4", "port": "7000"},
        {"host": "redis-node-5", "port": "7000"},
        {"host": "redis-node-6", "port": "7000"}
    ],
    "decode_responses": True,
    "skip_full_coverage_check": True,
    "max_connections_per_node": 100
}
```

#### 缓存预热策略
```python
async def warm_up_cache():
    """缓存预热"""
    # 预加载热点用户数据
    hot_users = await get_hot_users(limit=10000)

    tasks = []
    for user_id in hot_users:
        task = cache_user_info(user_id)
        tasks.append(task)

    # 并发预热
    await asyncio.gather(*tasks, return_exceptions=True)

async def cache_user_info(user_id: str):
    """缓存用户信息"""
    user_data = await get_user_from_db(user_id)
    if user_data:
        cache_key = f"user:{user_id}"
        await redis_client.setex(
            cache_key,
            3600,  # 1小时
            json.dumps(user_data.to_dict())
        )
```

### 3. 应用层优化

#### 异步处理
```python
class AsyncAuthenticationService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=100)
        self.semaphore = asyncio.Semaphore(1000)  # 限制并发数

    async def handle_auth_request(self, request: AuthRequest):
        async with self.semaphore:
            # CPU密集型任务使用线程池
            loop = asyncio.get_event_loop()

            # 异步执行密码验证
            password_valid = await loop.run_in_executor(
                self.executor,
                self.verify_password,
                request.password,
                request.stored_hash
            )

            if password_valid:
                # 生成令牌（CPU密集型）
                tokens = await loop.run_in_executor(
                    self.executor,
                    self.generate_tokens,
                    request.user_data
                )
                return tokens

            return None
```

## 监控和告警

### 1. 关键指标监控

#### Prometheus指标
```python
# 认证服务指标
auth_request_duration = Histogram(
    'auth_request_duration_seconds',
    'Time spent processing auth requests',
    ['endpoint', 'method', 'status']
)

auth_request_total = Counter(
    'auth_requests_total',
    'Total number of auth requests',
    ['endpoint', 'method', 'status']
)

active_sessions_gauge = Gauge(
    'auth_active_sessions',
    'Number of active user sessions'
)

token_validation_errors = Counter(
    'auth_token_validation_errors_total',
    'Total token validation errors',
    ['error_type']
)
```

#### Grafana仪表板
```json
{
  "dashboard": {
    "title": "认证服务监控",
    "panels": [
      {
        "title": "认证请求QPS",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(auth_requests_total[5m])",
            "legendFormat": "{{endpoint}} - {{method}}"
          }
        ]
      },
      {
        "title": "认证延迟",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(auth_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(auth_request_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "title": "活跃会话数",
        "type": "singlestat",
        "targets": [
          {
            "expr": "auth_active_sessions",
            "legendFormat": "Active Sessions"
          }
        ]
      }
    ]
  }
}
```

### 2. 告警规则

#### Prometheus告警
```yaml
groups:
- name: authentication_alerts
  rules:
  - alert: HighAuthErrorRate
    expr: rate(auth_requests_total{status=~"5.."}[5m]) / rate(auth_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "认证错误率过高"
      description: "认证错误率在过去5分钟内超过5%"

  - alert: HighAuthLatency
    expr: histogram_quantile(0.95, rate(auth_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "认证延迟过高"
      description: "95%的认证请求延迟超过2秒"

  - alert: TokenValidationErrors
    expr: rate(auth_token_validation_errors_total[5m]) > 10
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "令牌验证错误率过高"
      description: "令牌验证错误率过高，可能存在安全威胁"
```

## 灾难恢复

### 1. 数据备份策略

#### 自动备份脚本
```bash
#!/bin/bash
# auth_backup.sh

# 数据库备份
for i in {0..15}; do
    pg_dump -h user-db-$i -U app_user -d user_db_$i | \
    gzip > /backups/user_db_$i_$(date +%Y%m%d_%H%M%S).sql.gz
done

# Redis备份
redis-cli --rdb /backups/redis_$(date +%Y%m%d_%H%M%S).rdb

# 清理旧备份（保留7天）
find /backups -name "*.gz" -mtime +7 -delete
find /backups -name "*.rdb" -mtime +7 -delete
```

### 2. 故障转移

#### 自动故障转移
```python
class FailoverManager:
    def __init__(self):
        self.primary_services = []
        self.backup_services = []
        self.health_checker = HealthChecker()

    async def handle_service_failure(self, failed_service_url: str):
        """处理服务故障"""
        # 1. 将故障服务标记为不可用
        await self.mark_service_unhealthy(failed_service_url)

        # 2. 启动备用服务
        backup_service = await self.start_backup_service()

        # 3. 更新负载均衡器配置
        await self.update_load_balancer_config(failed_service_url, backup_service)

        # 4. 发送告警
        await self.send_failure_alert(failed_service_url, backup_service)

        # 5. 记录故障事件
        await self.log_failure_event(failed_service_url)

    async def start_backup_service(self) -> str:
        """启动备用服务"""
        # 从预分配的资源池中启动新实例
        new_service_url = f"http://backup-auth-{uuid.uuid4()[:8]}:8000"

        # 启动容器
        await self.start_service_container(new_service_url)

        # 等待服务就绪
        await self.wait_for_service_ready(new_service_url)

        return new_service_url
```

## 总结

本百万级用户认证架构通过以下关键技术和策略实现高性能和高可用性：

1. **水平扩展**: 无状态服务设计，支持动态扩缩容
2. **数据分片**: 用户数据按ID哈希分布到多个数据库分片
3. **多层缓存**: 内存缓存 + Redis集群 + 共享缓存的三级缓存架构
4. **负载均衡**: 智能负载均衡 + 健康检查 + 故障转移
5. **安全防护**: 速率限制、异常检测、令牌黑名单等多重安全机制
6. **监控告警**: 全方位的性能监控和实时告警系统
7. **容器化部署**: Docker + Kubernetes的现代化部署方案

该架构能够支持百万级并发用户，提供毫秒级的认证响应，同时保证系统的安全性和可靠性。