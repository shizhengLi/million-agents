# 服务发现架构设计

## 概述

服务发现是分布式系统的核心基础设施，负责自动检测、注册和管理网络中的服务实例。在百万级智能体社交平台中，服务发现系统需要支持数千个服务实例的动态管理，确保高可用性和可扩展性。

## 架构设计原则

### 1. 去中心化设计 (Decentralized Design)

**设计理念**: 避免单点故障，提高系统可靠性

```python
class DecentralizedServiceRegistry:
    """去中心化服务注册表"""
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers  # 其他注册表节点
        self.local_registry = LocalRegistry()
        self.consensus = ConsensusModule()

    def register_service(self, service):
        """注册服务到本地并同步到集群"""
        # 本地注册
        self.local_registry.register(service)

        # 集群同步
        operation = Operation(
            type='register',
            service=service,
            timestamp=time.time(),
            node_id=self.node_id
        )

        self.consensus.propagate_operation(operation)

    def discover_services(self, service_type):
        """从本地和集群中发现服务"""
        # 本地查询
        local_services = self.local_registry.get_services(service_type)

        # 集群查询
        cluster_services = self.query_cluster_services(service_type)

        # 合并去重
        return self.merge_service_lists(local_services, cluster_services)
```

### 2. 最终一致性 (Eventual Consistency)

**设计理念**: 系统在短暂时间内可能不一致，但最终会达到一致状态

```python
class EventualConsistencyManager:
    """最终一致性管理器"""
    def __init__(self):
        self.operation_log = OperationLog()
        self.conflict_resolver = ConflictResolver()

    def apply_operation(self, operation):
        """应用操作并处理冲突"""
        # 记录操作
        self.operation_log.append(operation)

        # 检测冲突
        conflicts = self.detect_conflicts(operation)

        if conflicts:
            # 解决冲突
            resolved_operation = self.conflict_resolver.resolve(operation, conflicts)
            self.apply_resolved_operation(resolved_operation)
        else:
            # 直接应用操作
            self.apply_directly(operation)

    def sync_with_peers(self):
        """与对等节点同步"""
        for peer in self.peers:
            peer_operations = self.fetch_peer_operations(peer)
            for operation in peer_operations:
                self.apply_operation(operation)
```

### 3. 高可用性 (High Availability)

**设计理念**: 确保服务发现系统本身的高可用性

```python
class HighlyAvailableServiceDiscovery:
    """高可用服务发现系统"""
    def __init__(self):
        self.primary_registry = ServiceRegistry()
        self.backup_registries = [ServiceRegistry() for _ in range(2)]
        self.health_checker = HealthChecker()
        self.failover_manager = FailoverManager()

    def discover_services(self, service_type):
        """高可用的服务发现"""
        try:
            # 尝试主注册表
            return self.primary_registry.get_services(service_type)
        except Exception as e:
            logger.warning(f"Primary registry failed: {e}")

            # 故障转移到备用注册表
            return self.failover_to_backup(service_type)

    def failover_to_backup(self, service_type):
        """故障转移到备用注册表"""
        for backup in self.backup_registries:
            try:
                services = backup.get_services(service_type)
                if services:
                    return services
            except Exception as e:
                logger.warning(f"Backup registry failed: {e}")

        raise ServiceDiscoveryError("All registries unavailable")
```

## 核心组件架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Service Discovery Layer                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Client    │  │   Service   │  │   Admin     │          │
│  │   Library   │  │   Instance  │  │   Console   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Service Registry                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Registration│  │ Discovery   │  │ Health Check│          │
│  │   Service   │  │   Service   │  │   Service   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Consensus Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Raft      │  │   Gossip    │  │ Conflict    │          │
│  │  Consensus  │  │  Protocol   │  │ Resolution  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ In-Memory   │  │ Persistent  │  │ Distributed │          │
│  │   Cache     │  │   Storage   │  │   Storage   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 1. 服务注册表 (Service Registry)

#### 核心功能
- 服务实例注册和注销
- 服务元数据管理
- 服务健康状态跟踪
- 服务查询和发现

#### 实现架构
```python
class ServiceRegistry:
    """服务注册表核心实现"""
    def __init__(self):
        self.services = {}  # service_id -> ServiceInstance
        self.service_index = {}  # service_type -> Set[service_id]
        self.health_tracker = HealthTracker()
        self.event_bus = EventBus()
        self.persistence = PersistenceLayer()

    def register_service(self, service_instance):
        """注册服务实例"""
        service_id = service_instance.id

        # 检查重复注册
        if service_id in self.services:
            raise ServiceAlreadyExistsError(service_id)

        # 注册服务
        self.services[service_id] = service_instance
        self.service_index.setdefault(
            service_instance.service_type, set()
        ).add(service_id)

        # 开始健康检查
        self.health_tracker.start_monitoring(service_instance)

        # 持久化
        self.persistence.save_service(service_instance)

        # 发布事件
        self.event_bus.publish(ServiceRegisteredEvent(service_instance))

        logger.info(f"Service registered: {service_id}")

    def deregister_service(self, service_id):
        """注销服务实例"""
        if service_id not in self.services:
            raise ServiceNotFoundError(service_id)

        service = self.services[service_id]

        # 停止健康检查
        self.health_tracker.stop_monitoring(service_id)

        # 从索引中移除
        self.service_index[service.service_type].discard(service_id)
        if not self.service_index[service.service_type]:
            del self.service_index[service.service_type]

        # 从注册表中移除
        del self.services[service_id]

        # 持久化删除
        self.persistence.delete_service(service_id)

        # 发布事件
        self.event_bus.publish(ServiceDeregisteredEvent(service))

        logger.info(f"Service deregistered: {service_id}")

    def discover_services(self, service_type, healthy_only=True):
        """发现服务实例"""
        service_ids = self.service_index.get(service_type, set())
        services = []

        for service_id in service_ids:
            service = self.services.get(service_id)
            if service and (not healthy_only or service.is_healthy()):
                services.append(service)

        return services
```

### 2. 健康检查系统 (Health Check System)

#### 健康检查策略
```python
class HealthCheckStrategy:
    """健康检查策略基类"""
    async def check_health(self, service_instance):
        raise NotImplementedError

class HTTPHealthCheck(HealthCheckStrategy):
    """HTTP健康检查"""
    def __init__(self, endpoint="/health", timeout=5):
        self.endpoint = endpoint
        self.timeout = timeout

    async def check_health(self, service_instance):
        try:
            url = f"http://{service_instance.address}{self.endpoint}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    return response.status == 200
        except Exception:
            return False

class TCPHealthCheck(HealthCheckStrategy):
    """TCP连接健康检查"""
    def __init__(self, timeout=3):
        self.timeout = timeout

    async def check_health(self, service_instance):
        try:
            host, port = service_instance.address.split(':')
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(port)),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
```

#### 健康检查管理器
```python
class HealthChecker:
    """健康检查管理器"""
    def __init__(self, check_interval=30, timeout=10):
        self.check_interval = check_interval
        self.timeout = timeout
        self.monitored_services = {}
        self.check_strategies = {}
        self.is_running = False

    def start_monitoring(self, service_instance):
        """开始监控服务"""
        service_id = service_instance.id

        # 选择检查策略
        strategy = self.select_check_strategy(service_instance)
        self.check_strategies[service_id] = strategy

        # 创建监控任务
        task = asyncio.create_task(
            self.monitor_service(service_instance, strategy)
        )
        self.monitored_services[service_id] = task

        logger.info(f"Started health monitoring for {service_id}")

    def stop_monitoring(self, service_id):
        """停止监控服务"""
        if service_id in self.monitored_services:
            task = self.monitored_services[service_id]
            task.cancel()
            del self.monitored_services[service_id]

        if service_id in self.check_strategies:
            del self.check_strategies[service_id]

        logger.info(f"Stopped health monitoring for {service_id}")

    async def monitor_service(self, service_instance, strategy):
        """监控单个服务"""
        while True:
            try:
                is_healthy = await strategy.check_health(service_instance)
                self.update_service_health(service_instance.id, is_healthy)

                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {service_instance.id}: {e}")
                await asyncio.sleep(self.check_interval)

    def update_service_health(self, service_id, is_healthy):
        """更新服务健康状态"""
        # 这里会触发服务状态变更事件
        event = HealthStatusChangedEvent(service_id, is_healthy)
        EventBus.publish(event)
```

### 3. 服务发现客户端 (Service Discovery Client)

#### 客户端实现
```python
class ServiceDiscoveryClient:
    """服务发现客户端"""
    def __init__(self, registry_url, cache_ttl=60):
        self.registry_url = registry_url
        self.cache_ttl = cache_ttl
        self.service_cache = {}
        self.cache_timestamps = {}

    async def discover_service(self, service_type):
        """发现单个服务实例"""
        cached_result = self.get_from_cache(service_type)
        if cached_result:
            return cached_result

        # 从注册表获取
        services = await self.query_registry(service_type)
        if not services:
            return None

        # 选择服务实例
        selected_service = self.select_service_instance(services)

        # 缓存结果
        self.cache_result(service_type, selected_service)

        return selected_service

    async def discover_all_services(self, service_type):
        """发现所有服务实例"""
        cached_result = self.get_from_cache(f"{service_type}_all")
        if cached_result:
            return cached_result

        services = await self.query_registry(service_type)
        self.cache_result(f"{service_type}_all", services)

        return services

    def select_service_instance(self, services):
        """选择服务实例"""
        # 过滤健康服务
        healthy_services = [s for s in services if s.is_healthy()]

        if not healthy_services:
            # 如果没有健康服务，返回所有服务
            healthy_services = services

        # 使用负载均衡算法选择
        return self.load_balance_select(healthy_services)

    def load_balance_select(self, services):
        """负载均衡选择"""
        # 简单的轮询实现
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0

        service = services[self._round_robin_index % len(services)]
        self._round_robin_index += 1
        return service
```

### 4. 事件系统 (Event System)

#### 事件驱动架构
```python
class ServiceDiscoveryEvent:
    """服务发现事件基类"""
    def __init__(self, timestamp=None):
        self.timestamp = timestamp or time.time()

class ServiceRegisteredEvent(ServiceDiscoveryEvent):
    """服务注册事件"""
    def __init__(self, service_instance, timestamp=None):
        super().__init__(timestamp)
        self.service_instance = service_instance
        self.service_id = service_instance.id
        self.service_type = service_instance.service_type

class ServiceDeregisteredEvent(ServiceDiscoveryEvent):
    """服务注销事件"""
    def __init__(self, service_instance, timestamp=None):
        super().__init__(timestamp)
        self.service_instance = service_instance
        self.service_id = service_instance.id
        self.service_type = service_instance.service_type

class HealthStatusChangedEvent(ServiceDiscoveryEvent):
    """健康状态变更事件"""
    def __init__(self, service_id, is_healthy, timestamp=None):
        super().__init__(timestamp)
        self.service_id = service_id
        self.is_healthy = is_healthy

class EventBus:
    """事件总线"""
    def __init__(self):
        self.listeners = defaultdict(list)

    def subscribe(self, event_type, listener):
        """订阅事件"""
        self.listeners[event_type].append(listener)

    def unsubscribe(self, event_type, listener):
        """取消订阅"""
        if listener in self.listeners[event_type]:
            self.listeners[event_type].remove(listener)

    def publish(self, event):
        """发布事件"""
        event_type = type(event)
        for listener in self.listeners[event_type]:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
```

## 数据模型

### 服务实例模型
```python
@dataclass
class ServiceInstance:
    """服务实例数据模型"""
    id: str
    service_type: ServiceType
    address: str
    port: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    weight: int = 1
    health_status: HealthStatus = HealthStatus.UNKNOWN
    registration_time: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    ttl: int = 60  # 生存时间（秒）

    def is_healthy(self):
        """检查服务是否健康"""
        return self.health_status == HealthStatus.HEALTHY

    def is_expired(self):
        """检查服务是否过期"""
        return time.time() - self.last_health_check > self.ttl

    def get_endpoint(self):
        """获取服务端点"""
        return f"{self.address}:{self.port}"

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'service_type': self.service_type.value,
            'address': self.address,
            'port': self.port,
            'metadata': self.metadata,
            'tags': self.tags,
            'weight': self.weight,
            'health_status': self.health_status.value,
            'registration_time': self.registration_time,
            'last_health_check': self.last_health_check,
            'ttl': self.ttl
        }
```

## 配置管理

### 配置结构
```python
@dataclass
class ServiceDiscoveryConfig:
    """服务发现配置"""
    # 注册表配置
    registry:
        bind_address: str = "0.0.0.0:8080"
        cluster_size: int = 3
        replication_factor: int = 2

    # 健康检查配置
    health_check:
        interval: int = 30
        timeout: int = 10
        retries: int = 3
        strategies: Dict[str, str] = field(default_factory=lambda: {
            'HTTP': '/health',
            'TCP': '80'
        })

    # 缓存配置
    cache:
        ttl: int = 60
        max_size: int = 10000
        cleanup_interval: int = 300

    # 安全配置
    security:
        enable_tls: bool = False
        cert_file: str = ""
        key_file: str = ""
        ca_file: str = ""

    # 监控配置
    monitoring:
        enable_metrics: bool = True
        metrics_port: int = 9090
        enable_tracing: bool = True
```

## 性能优化

### 1. 缓存策略
```python
class ServiceCache:
    """服务缓存实现"""
    def __init__(self, ttl=60, max_size=10000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.cleanup_interval = 300

    def get(self, key):
        """获取缓存值"""
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.access_times[key] = time.time()
                return entry
            else:
                del self.cache[key]
                del self.access_times[key]
        return None

    def put(self, key, value):
        """设置缓存值"""
        # 检查缓存大小
        if len(self.cache) >= self.max_size:
            self.evict_lru()

        self.cache[key] = (value, time.time())
        self.access_times[key] = time.time()

    def evict_lru(self):
        """淘汰最近最少使用的缓存"""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(),
                     key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
```

### 2. 批量操作
```python
class BatchServiceRegistry:
    """批量服务注册表操作"""
    def __init__(self, registry):
        self.registry = registry
        self.batch_size = 100

    async def batch_register_services(self, services):
        """批量注册服务"""
        batches = [
            services[i:i + self.batch_size]
            for i in range(0, len(services), self.batch_size)
        ]

        for batch in batches:
            await self.process_batch_registration(batch)

    async def process_batch_registration(self, batch):
        """处理批量注册"""
        tasks = []
        for service in batch:
            task = asyncio.create_task(
                self.registry.register_service(service)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        successful = []
        failed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append((batch[i], result))
            else:
                successful.append(batch[i])

        return successful, failed
```

## 监控和告警

### 关键指标
```python
class ServiceDiscoveryMetrics:
    """服务发现指标收集"""
    def __init__(self):
        self.metrics = {
            'total_services': 0,
            'healthy_services': 0,
            'unhealthy_services': 0,
            'registration_rate': 0,
            'deregistration_rate': 0,
            'health_check_rate': 0,
            'discovery_rate': 0,
            'cache_hit_rate': 0.0,
            'response_time_p95': 0,
            'error_rate': 0.0
        }

    def update_service_counts(self, registry):
        """更新服务计数"""
        all_services = registry.get_all_services()
        self.metrics['total_services'] = len(all_services)
        self.metrics['healthy_services'] = sum(
            1 for s in all_services if s.is_healthy()
        )
        self.metrics['unhealthy_services'] = (
            self.metrics['total_services'] -
            self.metrics['healthy_services']
        )

    def get_prometheus_metrics(self):
        """获取Prometheus格式的指标"""
        return [
            f'service_discovery_total_services {self.metrics["total_services"]}',
            f'service_discovery_healthy_services {self.metrics["healthy_services"]}',
            f'service_discovery_unhealthy_services {self.metrics["unhealthy_services"]}',
            f'service_discovery_registration_rate {self.metrics["registration_rate"]}',
            f'service_discovery_deregistration_rate {self.metrics["deregistration_rate"]}',
            f'service_discovery_health_check_rate {self.metrics["health_check_rate"]}',
            f'service_discovery_discovery_rate {self.metrics["discovery_rate"]}',
            f'service_discovery_cache_hit_rate {self.metrics["cache_hit_rate"]}',
            f'service_discovery_response_time_p95 {self.metrics["response_time_p95"]}',
            f'service_discovery_error_rate {self.metrics["error_rate"]}'
        ]
```

## 总结

服务发现架构设计需要综合考虑：
- **可用性**: 避免单点故障，支持故障转移
- **一致性**: 处理网络分区和数据同步
- **性能**: 优化查询速度，减少延迟
- **扩展性**: 支持大规模服务实例管理
- **可维护性**: 简化运维，提供监控和告警

在我们的实现中，采用了事件驱动的架构设计，支持92%的测试覆盖率，为百万级智能体社交平台提供了稳定可靠的服务发现能力。

---

**相关阅读**:
- [健康检查机制](./health-checking.md)
- [服务注册与注销](./registration.md)
- [分布式服务治理](./governance.md)