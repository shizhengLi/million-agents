# 大规模分布式系统可扩展性设计模式

## 概述

在设计百万级智能体社交平台时，我们需要考虑系统从1万用户到1000万用户的扩展路径。本文档详细介绍了大规模分布式系统的设计模式、架构演进策略和性能优化技术。

## 可扩展性架构演进

### 阶段一：单体应用 (1K - 10K 用户)

```
┌─────────────────────────────────────┐
│           Single Server             │
│  ┌─────────────────────────────┐    │
│  │        Application          │    │
│  │  ┌─────────┐ ┌─────────────┐ │    │
│  │  │ Web App │ │ Business    │ │    │
│  │  │         │ │ Logic       │ │    │
│  │  └─────────┘ └─────────────┘ │    │
│  │  ┌─────────┐ ┌─────────────┐ │    │
│  │  │ Auth    │ │ Database    │ │    │
│  │  │ Service │ │ (PostgreSQL)│ │    │
│  │  └─────────┘ └─────────────┘ │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │        Cache (Redis)        │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

**特点**:
- 简单部署和维护
- 快速开发和迭代
- 单点故障风险
- 扩展性受限

**实现示例**:
```python
# 单体应用架构
class MonolithicApplication:
    def __init__(self):
        self.database = PostgreSQLConnection()
        self.cache = RedisConnection()
        self.auth_service = AuthService()
        self.business_logic = BusinessLogic()

    async def handle_request(self, request):
        # 认证
        user = await self.auth_service.authenticate(request.token)

        # 业务逻辑
        result = await self.business_logic.process(request, user)

        return result

# 部署配置
app = MonolithicApplication()
server = HTTPServer(app)
server.run(host="0.0.0.0", port=8080)
```

### 阶段二：垂直拆分 (10K - 100K 用户)

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                           │
│                    (Nginx/HAProxy)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼────┐ ┌──────▼──────┐
│   Web Server  │ │  API   │ │   Admin     │
│   (Frontend)  │ │Server  │ │   Panel     │
└───────┬──────┘ └───┬────┘ └──────┬──────┘
        │            │            │
        └────────────┼────────────┘
                     │
    ┌────────────────▼────────────────┐
    │        Shared Services          │
    │  ┌─────────┐  ┌─────────────┐   │
    │  │  Auth   │  │    Cache    │   │
    │  │Service  │  │   (Redis)   │   │
    │  └─────────┘  └─────────────┘   │
    └────────────────┬────────────────┘
                     │
        ┌────────────▼────────────┐
        │     Database Cluster    │
        │   (Master-Slave)        │
        └─────────────────────────┘
```

**特点**:
- 服务按功能拆分
- 独立部署和扩展
- 共享数据库和缓存
- 初步的服务化

**实现示例**:
```python
# 微服务架构基础
class MicroserviceBase:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.registry = ServiceRegistry()
        self.health_checker = HealthChecker()

    async def start(self):
        # 注册服务
        await self.registry.register(
            service_name=self.service_name,
            address=f"{self.service_name}.local:8080",
            health_check_url="/health"
        )

        # 启动健康检查
        await self.health_checker.start()

    async def handle_request(self, request):
        # 通用请求处理逻辑
        pass

class AuthService(MicroserviceBase):
    def __init__(self):
        super().__init__("auth-service")
        self.database = PostgreSQLConnection()
        self.jwt_secret = "your-secret-key"

    async def authenticate(self, token: str):
        # JWT验证逻辑
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user = await self.database.get_user(payload["user_id"])
            return user
        except jwt.InvalidTokenError:
            return None

class APIService(MicroserviceBase):
    def __init__(self):
        super().__init__("api-service")
        self.auth_client = ServiceClient("auth-service")
        self.cache_client = RedisConnection()

    async def handle_request(self, request):
        # 验证用户身份
        user = await self.auth_client.authenticate(request.token)
        if not user:
            return {"error": "Unauthorized"}

        # 检查缓存
        cache_key = f"api:{request.path}:{user.id}"
        cached_result = await self.cache_client.get(cache_key)
        if cached_result:
            return cached_result

        # 处理业务逻辑
        result = await self.process_business_logic(request, user)

        # 缓存结果
        await self.cache_client.set(cache_key, result, ttl=300)

        return result
```

### 阶段三：水平拆分 (100K - 1M 用户)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CDN & Edge Cache                           │
│                    (CloudFlare/Akamai)                        │
└─────────────────────┬─────────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │    Global Load Balancer   │
        │     (Multi-Region)        │
        └───────┬───────┬───────┬───┘
                │       │       │
    ┌───────────▼───┐ ┌─▼─────────▼─┐
    │   Region A    │ │  Region B   │
    │   (US-East)   │ │ (EU-West)   │
    └───────┬───────┘ └───────┬─────┘
            │                 │
    ┌───────▼─────────────────▼───────┐
    │      Regional Load Balancer     │
    └───────┬─────────┬───────────────┘
            │         │
    ┌───────▼───┐ ┌───▼────────┐
    │ Service   │ │   Service  │
    │ Cluster A │ │  Cluster B │
    └───────┬───┘ └───────┬────┘
            │             │
    ┌───────▼─────────────▼───────┐
    │    Distributed Cache       │
    │    (Redis Cluster)         │
    └───────┬─────────┬──────────┘
            │         │
    ┌───────▼───┐ ┌───▼────────┐
    │ Database  │ │  Database  │
    │ Cluster A │ │ Cluster B  │
    │ (Sharded) │ │ (Sharded)  │
    └───────────┘ └────────────┘
```

**特点**:
- 多地域部署
- 数据库分片
- 分布式缓存
- 微服务网格

**实现示例**:
```python
# 分布式架构实现
class DistributedService:
    def __init__(self, service_name: str, region: str):
        self.service_name = service_name
        self.region = region
        self.service_mesh = ServiceMesh()
        self.distributed_cache = DistributedCache()
        self.sharded_database = ShardedDatabase()

    async def start(self):
        # 连接到服务网格
        await self.service_mesh.connect(
            service_name=self.service_name,
            region=self.region
        )

        # 初始化分布式缓存
        await self.distributed_cache.connect(
            nodes=await self.get_cache_nodes()
        )

        # 初始化分片数据库
        await self.sharded_database.connect(
            shards=await self.get_database_shards()
        )

class ShardedDatabase:
    def __init__(self):
        self.shards = {}
        self.sharding_strategy = ConsistentHashSharding()

    async def connect(self, shards: List[DatabaseShard]):
        for shard in shards:
            self.shards[shard.id] = DatabaseConnection(shard.config)

    async def get(self, key: str):
        shard_id = self.sharding_strategy.get_shard(key)
        shard = self.shards[shard_id]
        return await shard.get(key)

    async def set(self, key: str, value: any):
        shard_id = self.sharding_strategy.get_shard(key)
        shard = self.shards[shard_id]
        return await shard.set(key, value)

class ConsistentHashSharding:
    def __init__(self, virtual_nodes=150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []

    def add_shard(self, shard_id: str):
        for i in range(self.virtual_nodes):
            key = self.calculate_hash(f"{shard_id}:{i}")
            self.ring[key] = shard_id
        self.sorted_keys = sorted(self.ring.keys())

    def get_shard(self, key: str) -> str:
        if not self.ring:
            raise Exception("No shards available")

        hash_value = self.calculate_hash(key)

        for key in self.sorted_keys:
            if key >= hash_value:
                return self.ring[key]

        return self.ring[self.sorted_keys[0]]

    def calculate_hash(self, key: str) -> int:
        import hashlib
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class DistributedCache:
    def __init__(self):
        self.cache_nodes = {}
        self.consistent_hash = ConsistentHashSharding()

    async def connect(self, nodes: List[CacheNode]):
        for node in nodes:
            self.cache_nodes[node.id] = node
            self.consistent_hash.add_shard(node.id)

    async def get(self, key: str):
        node_id = self.consistent_hash.get_shard(key)
        node = self.cache_nodes[node_id]
        return await node.get(key)

    async def set(self, key: str, value: any, ttl: int = 3600):
        node_id = self.consistent_hash.get_shard(key)
        node = self.cache_nodes[node_id]
        return await node.set(key, value, ttl)

    async def invalidate(self, key: str):
        node_id = self.consistent_hash.get_shard(key)
        node = self.cache_nodes[node_id]
        return await node.delete(key)
```

### 阶段四：大规模分布式 (1M+ 用户)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Global Infrastructure                      │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Content       │    │   Edge          │    │   API       │ │
│  │   Delivery      │    │   Computing     │    │   Gateway   │ │
│  │   Network       │    │   (Lambda@Edge) │    │   (Kong)    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                      │                       │    │
│           └──────────────────────┼───────────────────────┘    │
│                                  │                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Service Mesh Layer                        │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │ │
│  │  │   Istio     │ │   Linkerd   │ │     Consul          │   │ │
│  │  │ Ingress     │ │ Sidecar     │ │ Service Discovery   │   │ │
│  │  │ Gateway     │ │ Proxy       │ │                      │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                  │                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Application Services                        │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │ │
│  │  │   Auth      │ │   Social    │ │     AI Agent        │   │ │
│  │  │   Service   │ │   Service   │ │     Service         │   │ │
│  │  │ (K8s Pods)  │ │ (K8s Pods)  │ │   (GPU Enabled)     │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                  │                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Data Layer                                 │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │ │
│  │  │   SQL       │ │   NoSQL     │ │     Time Series     │   │ │
│  │  │ (Citus)     │ │ (MongoDB)   │ │    (InfluxDB)       │   │ │
│  │  │   Cluster   │ │   Sharding  │ │                     │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**特点**:
- 全球多活架构
- 服务网格管理
- 多数据库混合使用
- 边缘计算支持
- 容器化和编排

**实现示例**:
```python
# 大规模分布式架构
class GlobalPlatform:
    def __init__(self):
        self.service_mesh = IstioServiceMesh()
        self.kubernetes_cluster = KubernetesCluster()
        self.global_database = GlobalDatabase()
        self.edge_computing = EdgeComputingPlatform()
        self.monitoring = DistributedMonitoring()

    async def deploy(self):
        # 部署服务网格
        await self.service_mesh.deploy()

        # 配置Kubernetes集群
        await self.kubernetes_cluster.configure_auto_scaling(
            min_replicas=10,
            max_replicas=1000,
            target_cpu_utilization=70
        )

        # 部署全球数据库
        await self.global_database.setup_multi_region_replication()

        # 配置边缘计算节点
        await self.edge_computing.deploy_edge_nodes()

        # 设置监控告警
        await self.monitoring.setup_global_monitoring()

class IstioServiceMesh:
    def __init__(self):
        self.pilot = Pilot()
        self.mixer = Mixer()
        self.citadel = Citadel()

    async def deploy(self):
        # 部署控制平面
        await self.pilot.deploy()
        await self.mixer.deploy()
        await self.citadel.deploy()

        # 配置流量管理
        await self.configure_traffic_management()

        # 配置安全策略
        await self.configure_security_policies()

    async def configure_traffic_management(self):
        # 配置虚拟服务
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {"name": "social-platform"},
            "spec": {
                "hosts": ["social-platform.com"],
                "http": [
                    {
                        "match": [{"uri": {"prefix": "/api/v1/auth"}}],
                        "route": [{"destination": {"host": "auth-service"}}],
                        "timeout": "5s"
                    },
                    {
                        "match": [{"uri": {"prefix": "/api/v1/social"}}],
                        "route": [{"destination": {"host": "social-service"}}],
                        "timeout": "10s",
                        "fault": {
                            "delay": {"percent": 0.1, "fixedDelay": "5s"}
                        }
                    }
                ]
            }
        }

        await self.apply_yaml(virtual_service)

class KubernetesCluster:
    def __init__(self):
        self.api_client = KubernetesApiClient()
        self.autoscaler = HorizontalPodAutoscaler()

    async def configure_auto_scaling(self, min_replicas: int,
                                   max_replicas: int,
                                   target_cpu_utilization: int):
        # 配置水平Pod自动扩展
        hpa_config = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": "social-platform-hpa"},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "social-platform"
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": target_cpu_utilization
                            }
                        }
                    }
                ]
            }
        }

        await self.api_client.create(hpa_config)

class GlobalDatabase:
    def __init__(self):
        self.primary_regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
        self.read_replicas = {}
        self.conflict_resolver = ConflictResolver()

    async def setup_multi_region_replication(self):
        for region in self.primary_regions:
            # 设置主数据库
            primary = await self.create_database_cluster(
                region=region,
                instance_type="db.r5.4xlarge",
                multi_az=True
            )

            # 设置只读副本
            for replica_region in self.get_replica_regions(region):
                replica = await self.create_read_replica(
                    primary=primary,
                    region=replica_region
                )
                self.read_replicas[replica_region] = replica

    async def write(self, table: str, data: dict, region: str = None):
        if region is None:
            region = self.get_optimal_write_region(data)

        primary = self.get_primary_database(region)
        result = await primary.write(table, data)

        # 异步复制到其他区域
        asyncio.create_task(self.replicate_to_other_regions(table, data, region))

        return result

    async def read(self, table: str, query: dict, region: str = None):
        if region is None:
            region = self.get_closest_region()

        # 优先从本地只读副本读取
        if region in self.read_replicas:
            replica = self.read_replicas[region]
            result = await replica.read(table, query)
            if result:
                return result

        # 回退到主数据库
        primary = self.get_primary_database(region)
        return await primary.read(table, query)

    async def replicate_to_other_regions(self, table: str, data: dict,
                                       source_region: str):
        for target_region in self.primary_regions:
            if target_region != source_region:
                try:
                    target_primary = self.get_primary_database(target_region)
                    await target_primary.write(table, data)
                except Exception as e:
                    print(f"Replication failed to {target_region}: {e}")

class EdgeComputingPlatform:
    def __init__(self):
        self.edge_nodes = {}
        self.cloudflare_api = CloudflareAPI()
        self.aws_lambda_edge = AWSLambdaEdge()

    async def deploy_edge_nodes(self):
        # 部署Cloudflare Workers
        await self.cloudflare_api.deploy_worker(
            name="auth-validator",
            script=self.get_auth_validator_script()
        )

        # 部署AWS Lambda@Edge函数
        await self.aws_lambda_edge.deploy_function(
            name="response-cache",
            regions=["us-east-1"],
            trigger="origin-response"
        )

    def get_auth_validator_script(self) -> str:
        return """
        addEventListener('fetch', event => {
            event.respondWith(handleRequest(event.request))
        })

        async function handleRequest(request) {
            // 快速认证验证
            const authHeader = request.headers.get('Authorization')
            if (!authHeader) {
                return new Response('Unauthorized', { status: 401 })
            }

            // 验证JWT token
            const token = authHeader.replace('Bearer ', '')
            const isValid = await validateJWT(token)

            if (!isValid) {
                return new Response('Invalid token', { status: 401 })
            }

            // 转发到源站
            return fetch(request)
        }

        async function validateJWT(token) {
            // 简化的JWT验证逻辑
            try {
                const parts = token.split('.')
                if (parts.length !== 3) return false

                const payload = JSON.parse(atob(parts[1]))
                return payload.exp > Date.now() / 1000
            } catch {
                return false
            }
        }
        """
```

## 性能优化策略

### 1. 数据库优化

```python
class DatabaseOptimizer:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.index_manager = IndexManager()
        self.connection_pool = ConnectionPool()

    async def optimize_queries(self):
        # 分析慢查询
        slow_queries = await self.query_analyzer.get_slow_queries(
            min_duration=1000  # 超过1秒的查询
        )

        for query in slow_queries:
            # 生成优化建议
            suggestions = await self.query_analyzer.suggest_optimizations(query)

            # 自动创建索引
            if suggestions.should_create_index:
                await self.index_manager.create_index(
                    table=query.table,
                    columns=suggested.columns,
                    index_type="btree"
                )

    async def implement_read_write_splitting(self):
        # 读写分离配置
        read_config = {
            "read_replicas": [
                {"host": "read-replica-1", "weight": 1},
                {"host": "read-replica-2", "weight": 1},
                {"host": "read-replica-3", "weight": 2}
            ],
            "routing_strategy": "weighted_round_robin",
            "failover_enabled": True
        }

        self.connection_pool.configure_read_write_splitting(read_config)

class QueryAnalyzer:
    async def get_slow_queries(self, min_duration: int) -> List[Query]:
        # 从数据库获取慢查询日志
        query = """
        SELECT query, calls, total_time, mean_time, rows
        FROM pg_stat_statements
        WHERE mean_time > %s
        ORDER BY total_time DESC
        LIMIT 100
        """
        return await self.database.execute(query, [min_duration])

    async def suggest_optimizations(self, query: Query) -> OptimizationSuggestion:
        suggestions = OptimizationSuggestion()

        # 检查是否缺少索引
        if self.detect_missing_index(query):
            suggestions.should_create_index = True
            suggested.columns = self.extract_index_columns(query)

        # 检查是否需要重写查询
        if self.detect_inefficient_join(query):
            suggestions.should_rewrite = True
            suggestions.rewrite_strategy = "use_subquery_instead"

        # 检查是否可以添加缓存
        if self.detect_cacheable_query(query):
            suggestions.should_cache = True
            suggestions.cache_ttl = self.calculate_cache_ttl(query)

        return suggestions
```

### 2. 缓存策略优化

```python
class AdvancedCacheManager:
    def __init__(self):
        self.cache_layers = {
            "l1": MemoryCache(max_size=1000),      # 本地内存缓存
            "l2": RedisCache(cluster="redis-l2"), # 分布式Redis缓存
            "l3": CloudCache(provider="aws")       # 云端缓存
        }
        self.cache_warmer = CacheWarmer()
        self.invalidation_manager = CacheInvalidationManager()

    async def get(self, key: str) -> any:
        # 多层缓存查询
        for layer_name, cache in self.cache_layers.items():
            try:
                value = await cache.get(key)
                if value is not None:
                    # 回填到上层缓存
                    await self.backfill_to_upper_layers(layer_name, key, value)
                    return value
            except Exception as e:
                print(f"Cache {layer_name} error: {e}")
                continue

        return None

    async def set(self, key: str, value: any, ttl: int = None):
        # 写入所有缓存层
        tasks = []
        for cache in self.cache_layers.values():
            task = asyncio.create_task(cache.set(key, value, ttl))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def backfill_to_upper_layers(self, current_layer: str, key: str, value: any):
        # 回填到上层缓存
        layers = list(self.cache_layers.keys())
        current_index = layers.index(current_layer)

        for i in range(current_index):
            upper_layer = layers[i]
            cache = self.cache_layers[upper_layer]
            try:
                await cache.set(key, value)
            except Exception as e:
                print(f"Backfill to {upper_layer} failed: {e}")

class CacheWarmer:
    def __init__(self):
        self.warming_schedule = {}
        self.access_analyzer = AccessPatternAnalyzer()

    async def schedule_cache_warming(self, data_source: str,
                                   query_pattern: str,
                                   schedule: str):
        # 定期预热热点数据
        self.warming_schedule[data_source] = {
            "query_pattern": query_pattern,
            "schedule": schedule,
            "last_warmed": 0
        }

        asyncio.create_task(self.warming_loop())

    async def warming_loop(self):
        while True:
            try:
                current_time = time.time()

                for data_source, config in self.warming_schedule.items():
                    if self.should_warm(config, current_time):
                        await self.warm_cache(data_source, config)
                        config["last_warmed"] = current_time

                await asyncio.sleep(60)  # 每分钟检查一次

            except Exception as e:
                print(f"Cache warming error: {e}")
                await asyncio.sleep(60)

    async def warm_cache(self, data_source: str, config: dict):
        # 预热缓存
        hot_keys = await self.get_hot_keys(data_source, config["query_pattern"])

        cache_manager = AdvancedCacheManager()
        for key in hot_keys:
            value = await self.get_data_from_source(data_source, key)
            await cache_manager.set(key, value, ttl=3600)

class CacheInvalidationManager:
    def __init__(self):
        self.invalidation_rules = {}
        self.message_broker = MessageBroker()

    async def register_invalidation_rule(self, table: str,
                                       cache_key_pattern: str,
                                       events: List[str]):
        # 注册缓存失效规则
        if table not in self.invalidation_rules:
            self.invalidation_rules[table] = []

        self.invalidation_rules[table].append({
            "cache_key_pattern": cache_key_pattern,
            "events": events
        })

        # 订阅数据库变更事件
        for event in events:
            await self.message_broker.subscribe(f"db.{table}.{event}",
                                              self.handle_database_change)

    async def handle_database_change(self, event_data: dict):
        # 处理数据库变更事件
        table = event_data["table"]
        operation = event_data["operation"]
        changed_data = event_data["data"]

        if table in self.invalidation_rules:
            for rule in self.invalidation_rules[table]:
                if operation in rule["events"]:
                    cache_keys = self.generate_cache_keys(
                        rule["cache_key_pattern"],
                        changed_data
                    )
                    await self.invalidate_cache_keys(cache_keys)

    async def invalidate_cache_keys(self, cache_keys: List[str]):
        cache_manager = AdvancedCacheManager()
        for key in cache_keys:
            await cache_manager.delete(key)
```

## 容器化和编排

### Kubernetes部署配置

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: social-platform

---
# auth-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
  namespace: social-platform
spec:
  replicas: 3
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
        image: social-platform/auth-service:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# auth-service-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: auth-service-hpa
  namespace: social-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: auth-service
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# auth-service-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: auth-service
  namespace: social-platform
spec:
  selector:
    app: auth-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: social-platform-ingress
  namespace: social-platform
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.social-platform.com
    secretName: social-platform-tls
  rules:
  - host: api.social-platform.com
    http:
      paths:
      - path: /api/v1/auth
        pathType: Prefix
        backend:
          service:
            name: auth-service
            port:
              number: 80
      - path: /api/v1/social
        pathType: Prefix
        backend:
          service:
            name: social-service
            port:
              number: 80
```

## 监控和可观测性

### Prometheus + Grafana监控

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - "social-platform-rules.yml"

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093

    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name

      - job_name: 'redis'
        static_configs:
          - targets: ['redis-exporter:9121']

      - job_name: 'postgres'
        static_configs:
          - targets: ['postgres-exporter:9187']

  social-platform-rules.yml: |
    groups:
      - name: social-platform
        rules:
          - alert: HighErrorRate
            expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"
              description: "Error rate is {{ $value | humanizePercentage }}"

          - alert: HighResponseTime
            expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High response time detected"
              description: "95th percentile response time is {{ $value }}s"

          - alert: DatabaseConnectionsHigh
            expr: pg_stat_activity_count > 80
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High database connections"
              description: "Database has {{ $value }} active connections"

          - alert: RedisMemoryHigh
            expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "Redis memory usage high"
              description: "Redis memory usage is {{ $value | humanizePercentage }}"
```

## 总结

大规模分布式系统的设计需要考虑：

1. **渐进式架构演进**: 从单体到微服务，从单区域到多区域
2. **数据分片策略**: 水平分片、垂直分片、地理分片
3. **缓存架构**: 多层缓存、预热策略、失效机制
4. **容器化部署**: Kubernetes自动扩缩容、服务网格
5. **监控可观测**: 全链路追踪、指标监控、告警系统

通过这些设计模式和最佳实践，我们可以构建一个能够支撑千万级用户的分布式系统架构。

---

**相关阅读**:
- [分布式系统核心概念](../knowledge-base/core-concepts.md)
- [问题解决方案](../problem-solving/challenges-solutions.md)
- [负载均衡器设计原理](../load-balancer/design-principles.md)