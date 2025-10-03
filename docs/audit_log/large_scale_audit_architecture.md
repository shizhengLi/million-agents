# 大规模审计日志系统架构设计

## 概述

本文档详细介绍了支持百万级用户的分布式审计日志系统架构设计，包括高可用性、高性能、可扩展性的技术方案和最佳实践。

## 架构设计原则

### 1. **可扩展性（Scalability）**
- 水平扩展能力
- 分片和分区策略
- 负载均衡机制

### 2. **高可用性（High Availability）**
- 多副本数据存储
- 故障自动转移
- 零停机维护

### 3. **高性能（High Performance）**
- 低延迟写入
- 快速查询响应
- 批量处理优化

### 4. **数据一致性（Data Consistency）**
- 最终一致性模型
- 事务完整性保证
- 数据完整性验证

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  Web App  │  │  Mobile   │  │   API     │  │ Micro     │   │
│  │  Service  │  │  Client   │  │ Gateway   │  │ Services  │   │
│  └─────────┬─┘  └─────────┬─┘  └─────────┬─┘  └─────────┬─┘   │
└────────────┼────────────────┼────────────────┼─────────────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Audit SDK Layer  │
                    │                   │
                    │ • Event Capture  │
                    │ • Validation      │
                    │ • Enrichment      │
                    │ • Buffering       │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Message Queue   │
                    │   (Kafka Cluster) │
                    │                   │
                    │ • High Throughput │
                    │ • Durability      │
                    │ • Ordering        │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼─────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│  Ingestion      │  │  Processing     │  │  Storage        │
│  Service        │  │  Engine         │  │  Cluster        │
│                 │  │                 │  │                 │
│ • Load Balance  │  │ • Real-time     │  │ • Hot Storage   │
│ • Rate Limiting │  │ • Batch         │  │ • Warm Storage  │
│ • Validation    │  │ • ML Analysis   │  │ • Cold Archive  │
│ • Transformation│  │ • Alerting      │  │ • Backup        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Query & Search  │
                    │  Service         │
                    │                   │
                    │ • API Gateway   │
                    │ • Elasticsearch  │
                    │ • Analytics      │
                    │ • Dashboard      │
                    └─────────────────┘
```

## 核心组件详细设计

### 1. **Audit SDK层**

#### 设计目标
- 统一的事件收集接口
- 最小化性能影响
- 异步批量处理

#### 技术实现
```python
class AuditSDK:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.buffer = CircularBuffer(max_size=config.buffer_size)
        self.producer = KafkaProducer(**config.kafka_config)
        self.executor = ThreadPoolExecutor(max_workers=config.worker_threads)
        self.metadata_enricher = MetadataEnricher()

    async def capture_event(self, event: AuditEvent):
        """捕获审计事件"""
        try:
            # 事件验证
            self.validate_event(event)

            # 元数据增强
            enriched_event = await self.metadata_enricher.enrich(event)

            # 添加到缓冲区
            self.buffer.put(enriched_event)

            # 异步发送
            if self.buffer.is_full() or self.buffer.time_since_last_flush > self.config.flush_interval:
                self.flush_buffer()

        except Exception as e:
            # 本地降级处理
            self.fallback_handler.handle(event, e)

    def flush_buffer(self):
        """批量发送缓冲区事件"""
        events = self.buffer.flush()
        if events:
            future = self.producer.send_batch(events)
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)

class MetadataEnricher:
    """元数据增强器"""
    def __init__(self):
        self.geoip_resolver = GeoIPResolver()
        self.device_detector = DeviceDetector()

    async def enrich(self, event: AuditEvent) -> AuditEvent:
        """增强事件元数据"""
        enriched_data = event.additional_data.copy()

        # 地理位置增强
        if event.ip_address:
            geo_info = await self.geoip_resolver.lookup(event.ip_address)
            enriched_data['geo_location'] = geo_info

        # 设备信息增强
        if event.user_agent:
            device_info = self.device_detector.parse(event.user_agent)
            enriched_data['device_info'] = device_info

        # 会话信息增强
        if hasattr(event, 'session_id'):
            session_info = await self.get_session_info(event.session_id)
            enriched_data['session_info'] = session_info

        return AuditEvent(
            event_id=event.event_id,
            user_id=event.user_id,
            action=event.action,
            resource=event.resource,
            severity=event.severity,
            additional_data=enriched_data,
            timestamp=event.timestamp
        )
```

### 2. **消息队列层**

#### Kafka集群配置
```yaml
# Kafka集群配置
kafka:
  brokers:
    - kafka-1:9092
    - kafka-2:9092
    - kafka-3:9092

  topics:
    audit-events:
      partitions: 12
      replication_factor: 3
      retention_hours: 720  # 30天
      segment_size: 1073741824  # 1GB

    audit-events-high-priority:
      partitions: 6
      replication_factor: 3
      retention_hours: 2160  # 90天
      cleanup_policy: compact

  producer:
    acks: all
    retries: 3
    batch_size: 16384
    linger_ms: 10
    compression_type: lz4
    max_in_flight_requests_per_connection: 5

  consumer:
    group_id: audit-processor
    auto_offset_reset: earliest
    enable_auto_commit: false
    max_poll_records: 1000
    fetch_min_bytes: 1024
    fetch_max_wait_ms: 500
```

#### 分区策略
```python
class AuditEventPartitioner:
    """审计事件分区策略"""

    def __init__(self):
        self.user_partitions = 8
        self.resource_partitions = 4

    def partition(self, event: AuditEvent, num_partitions: int) -> int:
        """根据事件内容选择分区"""
        # 高优先级事件使用特定分区
        if event.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]:
            return self._hash_string(f"{event.user_id}:high") % num_partitions

        # 按用户ID分区（保证用户事件的顺序性）
        user_partition = self._hash_string(event.user_id) % self.user_partitions

        # 按资源类型分区
        resource_partition = self._hash_string(event.resource) % self.resource_partitions

        # 组合分区
        return (user_partition + resource_partition) % num_partitions

    def _hash_string(self, s: str) -> int:
        """字符串哈希"""
        return hashlib.md5(s.encode()).hexdigest()
```

### 3. **处理引擎层**

#### 流处理架构
```python
class AuditStreamProcessor:
    """审计流处理器"""

    def __init__(self):
        self.kafka_consumer = KafkaConsumer(**self.consumer_config)
        self.processors = {
            'real_time': RealTimeProcessor(),
            'batch': BatchProcessor(),
            'ml_analysis': MLAnalysisProcessor(),
            'alerting': AlertingProcessor()
        }
        self.state_store = RedisStateStore()

    def start_processing(self):
        """启动流处理"""
        for message in self.kafka_consumer:
            try:
                event_data = json.loads(message.value)
                event = AuditEvent.from_dict(event_data)

                # 并行处理
                futures = []
                for processor_name, processor in self.processors.items():
                    future = self.executor.submit(processor.process, event)
                    futures.append((processor_name, future))

                # 等待处理完成并处理异常
                for processor_name, future in futures:
                    try:
                        future.result(timeout=30)
                    except Exception as e:
                        self.handle_processing_error(processor_name, event, e)

                # 提交偏移量
                self.kafka_consumer.commit()

            except Exception as e:
                self.handle_message_error(message, e)

class RealTimeProcessor:
    """实时处理器"""

    def process(self, event: AuditEvent):
        """实时处理事件"""
        # 实时统计更新
        self.update_real_time_stats(event)

        # 会话跟踪
        self.update_session_tracking(event)

        # 异常检测
        anomalies = self.detect_real_time_anomalies(event)
        if anomalies:
            self.trigger_immediate_alerts(anomalies)

        # 索引更新
        self.update_search_index(event)

class BatchProcessor:
    """批量处理器"""

    def __init__(self):
        self.batch_window = timedelta(minutes=5)
        self.batch_size = 1000

    def process(self, event: AuditEvent):
        """批量处理事件"""
        # 添加到时间窗口
        self.add_to_time_window(event)

        # 检查是否需要处理批次
        if self.should_process_batch():
            self.process_batch()

    def process_batch(self):
        """处理批次数据"""
        batch = self.get_current_batch()

        # 聚合统计
        self.compute_aggregations(batch)

        # 趋势分析
        self.analyze_trends(batch)

        # 数据归档
        self.archive_batch(batch)

        # 清理已处理数据
        self.clear_processed_batch()
```

### 4. **存储层设计**

#### 分层存储架构
```python
class TieredAuditStorage:
    """分层审计存储"""

    def __init__(self):
        self.hot_storage = ElasticsearchCluster()  # 热数据：30天
        self.warm_storage = ClickHouseCluster()    # 温数据：1年
        self.cold_storage = S3Archive()            # 冷数据：7年

    def store_event(self, event: AuditEvent):
        """存储事件"""
        # 根据时间选择存储层
        age = datetime.now(timezone.utc) - event.timestamp

        if age.days <= 30:
            self.hot_storage.index(event)
        elif age.days <= 365:
            self.warm_storage.insert(event)
        else:
            self.cold_storage.archive(event)

    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """查询事件"""
        # 根据查询时间范围选择存储层
        time_range = query.time_range

        if time_range.end - time_range.start <= timedelta(days=30):
            return self.hot_storage.search(query)
        elif time_range.end - time_range.start <= timedelta(days=365):
            return self.warm_storage.query(query)
        else:
            # 跨层查询
            results = []
            results.extend(self.hot_storage.search(query))
            results.extend(self.warm_storage.query(query))
            results.extend(self.cold_storage.search(query))
            return sorted(results, key=lambda x: x.timestamp)
```

#### Elasticsearch集群配置
```yaml
elasticsearch:
  cluster:
    name: audit-cluster
    nodes:
      - host: es-1
        port: 9200
        roles: [master, data, ingest]
      - host: es-2
        port: 9200
        roles: [master, data, ingest]
      - host: es-3
        port: 9200
        roles: [master, data, ingest]

  index_templates:
    audit-events:
      settings:
        number_of_shards: 12
        number_of_replicas: 2
        refresh_interval: 5s
        max_result_window: 50000

      mappings:
        properties:
          timestamp:
            type: date
            format: strict_date_optional_time||epoch_millis

          user_id:
            type: keyword
            doc_values: true

          action:
            type: keyword
            doc_values: true

          resource:
            type: keyword
            doc_values: true

          severity:
            type: keyword
            doc_values: true

          ip_address:
            type: ip
            doc_values: true

          geo_location:
            type: geo_point

          additional_data:
            type: object
            dynamic: true

          @timestamp:
            type: date
```

#### ClickHouse配置
```sql
-- 创建审计日志表
CREATE TABLE audit_events (
    event_id String,
    timestamp DateTime64(3),
    user_id String,
    action String,
    resource String,
    severity Enum8('info' = 1, 'warning' = 2, 'high' = 3, 'critical' = 4),
    status String,
    ip_address IPv4,
    user_agent String,
    additional_data String,
    service_name String,
    created_date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, timestamp, action)
TTL timestamp + toIntervalYear(1);

-- 创建物化视图用于实时统计
CREATE MATERIALIZED VIEW audit_events_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, action, toHour(timestamp))
AS SELECT
    user_id,
    action,
    toHour(timestamp) as hour,
    count() as event_count
FROM audit_events
GROUP BY user_id, action, hour;
```

### 5. **查询和搜索服务**

#### API网关设计
```python
class AuditQueryAPI:
    """审计查询API"""

    def __init__(self):
        self.search_service = ElasticsearchService()
        self.analytics_service = ClickHouseService()
        self.cache = RedisCache()
        self.rate_limiter = RateLimiter()

    async def search_events(self, request: SearchRequest) -> SearchResponse:
        """搜索审计事件"""
        # 限流检查
        if not await self.rate_limiter.check_limit(request.user_id):
            raise RateLimitExceeded()

        # 缓存检查
        cache_key = self.generate_cache_key(request)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return SearchResponse.from_json(cached_result)

        # 执行搜索
        if request.time_range.days <= 30:
            # 使用Elasticsearch
            results = await self.search_service.search(request)
        else:
            # 使用ClickHouse
            results = await self.analytics_service.query(request)

        # 缓存结果
        await self.cache.set(cache_key, results.to_json(), ttl=300)

        return results

    async def get_statistics(self, request: StatsRequest) -> StatsResponse:
        """获取统计信息"""
        # 预聚合数据查询
        if request.granularity == 'hour':
            return await self.analytics_service.get_hourly_stats(request)
        elif request.granularity == 'day':
            return await self.analytics_service.get_daily_stats(request)
        else:
            return await self.analytics_service.get_monthly_stats(request)

    async def export_data(self, request: ExportRequest) -> str:
        """导出数据"""
        # 异步导出任务
        task_id = str(uuid.uuid4())

        # 提交导出任务
        await self.export_queue.submit({
            'task_id': task_id,
            'request': request.to_dict(),
            'user_id': request.user_id
        })

        return task_id
```

#### 实时仪表板
```python
class AuditDashboard:
    """审计仪表板"""

    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.metrics_collector = MetricsCollector()

    async def stream_real_time_metrics(self, websocket: WebSocket):
        """实时指标流"""
        await websocket.accept()

        try:
            while True:
                # 收集实时指标
                metrics = await self.metrics_collector.get_current_metrics()

                # 发送到前端
                await websocket.send_json({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })

                # 等待下一次更新
                await asyncio.sleep(1)

        except WebSocketDisconnect:
            pass

    async def get_dashboard_data(self, filters: Dict) -> Dict:
        """获取仪表板数据"""
        # 并行获取各种数据
        tasks = [
            self.get_event_overview(filters),
            self.get_top_users(filters),
            self.get_security_alerts(filters),
            self.get_system_health(),
            self.get_compliance_status(filters)
        ]

        results = await asyncio.gather(*tasks)

        return {
            'overview': results[0],
            'top_users': results[1],
            'security_alerts': results[2],
            'system_health': results[3],
            'compliance': results[4]
        }
```

## 性能优化策略

### 1. **写入性能优化**

#### 批量写入策略
```python
class BatchWriteOptimizer:
    def __init__(self):
        self.write_buffer = []
        self.buffer_size = 1000
        self.flush_interval = 5
        self.last_flush = time.time()

    async def add_event(self, event: AuditEvent):
        """添加事件到缓冲区"""
        self.write_buffer.append(event)

        # 检查是否需要刷新
        if (len(self.write_buffer) >= self.buffer_size or
            time.time() - self.last_flush > self.flush_interval):
            await self.flush_events()

    async def flush_events(self):
        """批量写入事件"""
        if not self.write_buffer:
            return

        try:
            # 批量写入Elasticsearch
            await self.bulk_index_events(self.write_buffer)

            # 批量写入ClickHouse
            await self.bulk_insert_events(self.write_buffer)

            self.write_buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            # 写入失败时的重试策略
            await self.retry_failed_events(self.write_buffer)

    async def bulk_index_events(self, events: List[AuditEvent]):
        """批量索引到Elasticsearch"""
        bulk_body = []
        for event in events:
            bulk_body.append({
                "index": {"_index": f"audit-events-{event.timestamp.strftime('%Y-%m')}"}
            })
            bulk_body.append(event.to_dict())

        await self.es_client.bulk(body=bulk_body)
```

#### 异步写入优化
```python
class AsyncAuditWriter:
    def __init__(self):
        self.write_queue = asyncio.Queue(maxsize=10000)
        self.workers = [asyncio.create_task(self._worker()) for _ in range(10)]

    async def _worker(self):
        """异步写入工作者"""
        while True:
            try:
                events = await self._get_batch()
                await self._process_batch(events)
            except Exception as e:
                logger.error(f"Write worker error: {e}")

    async def _get_batch(self) -> List[AuditEvent]:
        """获取批量事件"""
        events = []
        try:
            # 等待第一个事件
            first_event = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
            events.append(first_event)

            # 收集更多事件（非阻塞）
            while len(events) < 100:
                try:
                    event = self.write_queue.get_nowait()
                    events.append(event)
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            pass

        return events
```

### 2. **查询性能优化**

#### 多级缓存策略
```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        self.l2_cache = RedisCache()  # Redis缓存
        self.l3_cache = MemcachedCache()  # 分布式缓存

    async def get(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        # L1缓存
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2缓存
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value

        # L3缓存
        value = await self.l3_cache.get(key)
        if value:
            await self.l2_cache.set(key, value, ttl=3600)
            self.l1_cache[key] = value
            return value

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """多级缓存设置"""
        self.l1_cache[key] = value
        await self.l2_cache.set(key, value, ttl=ttl)
        await self.l3_cache.set(key, value, ttl=ttl)
```

#### 索引优化策略
```python
class IndexOptimizer:
    def __init__(self):
        self.elastic_client = ElasticsearchClient()

    async def optimize_indices(self):
        """优化Elasticsearch索引"""
        # 强制合并段
        await self.elastic_client.indices.forge_merge(
            index="audit-events-*",
            max_num_segments=1
        )

        # 刷新索引
        await self.elastic_client.indices.refresh(
            index="audit-events-*"
        )

        # 更新映射
        await self.update_mapping_template()

    async def update_mapping_template(self):
        """更新映射模板"""
        template = {
            "index_patterns": ["audit-events-*"],
            "template": {
                "settings": {
                    "number_of_shards": 12,
                    "number_of_replicas": 2,
                    "refresh_interval": "30s",
                    "index.codec": "best_compression"
                },
                "mappings": {
                    "properties": {
                        "timestamp": {
                            "type": "date",
                            "format": "strict_date_optional_time||epoch_millis"
                        },
                        "user_id": {
                            "type": "keyword",
                            "doc_values": true,
                            "ignore_above": 256
                        },
                        "action": {
                            "type": "keyword",
                            "doc_values": true
                        }
                    }
                }
            }
        }

        await self.elastic_client.indices.put_index_template(
            name="audit-events-template",
            body=template
        )
```

## 监控和运维

### 1. **系统监控**

#### 关键指标监控
```python
class AuditSystemMonitor:
    def __init__(self):
        self.metrics = {
            'ingestion_rate': Gauge('audit_ingestion_rate', 'Events per second'),
            'processing_latency': Histogram('audit_processing_latency', 'Processing latency'),
            'storage_usage': Gauge('audit_storage_usage', 'Storage usage'),
            'error_rate': Gauge('audit_error_rate', 'Error rate'),
            'query_latency': Histogram('audit_query_latency', 'Query latency')
        }

    async def collect_metrics(self):
        """收集系统指标"""
        while True:
            # 收集各组件指标
            ingestion_metrics = await self.get_ingestion_metrics()
            storage_metrics = await self.get_storage_metrics()
            query_metrics = await self.get_query_metrics()

            # 更新Prometheus指标
            self.metrics['ingestion_rate'].set(ingestion_metrics['events_per_second'])
            self.metrics['storage_usage'].set(storage_metrics['usage_bytes'])
            self.metrics['error_rate'].set(ingestion_metrics['error_rate'])

            await asyncio.sleep(10)

    async def get_ingestion_metrics(self) -> Dict:
        """获取接入层指标"""
        return {
            'events_per_second': await self.get_kafka_throughput(),
            'lag': await self.get_consumer_lag(),
            'error_rate': await self.get_error_rate()
        }
```

#### 健康检查
```python
class HealthCheckService:
    def __init__(self):
        self.components = {
            'kafka': KafkaHealthChecker(),
            'elasticsearch': ElasticsearchHealthChecker(),
            'clickhouse': ClickHouseHealthChecker(),
            'redis': RedisHealthChecker()
        }

    async def check_health(self) -> Dict:
        """系统健康检查"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }

        overall_healthy = True

        for component_name, checker in self.components.items():
            try:
                component_health = await checker.check()
                health_status['components'][component_name] = component_health

                if component_health['status'] != 'healthy':
                    overall_healthy = False

            except Exception as e:
                health_status['components'][component_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                overall_healthy = False

        if not overall_healthy:
            health_status['status'] = 'unhealthy'

        return health_status
```

### 2. **故障处理**

#### 自动故障转移
```python
class FailoverManager:
    def __init__(self):
        self.primary_components = {}
        self.backup_components = {}
        self.health_checker = HealthCheckService()

    async def monitor_and_failover(self):
        """监控并自动故障转移"""
        while True:
            health_status = await self.health_checker.check_health()

            for component, status in health_status['components'].items():
                if status['status'] == 'unhealthy':
                    await self.failover_component(component)

            await asyncio.sleep(30)

    async def failover_component(self, component_name: str):
        """组件故障转移"""
        logger.warning(f"Initiating failover for {component_name}")

        if component_name == 'elasticsearch':
            await self.failover_elasticsearch()
        elif component_name == 'clickhouse':
            await self.failover_clickhouse()
        elif component_name == 'kafka':
            await self.failover_kafka()

        # 发送告警
        await self.send_failover_alert(component_name)

    async def failover_elasticsearch(self):
        """Elasticsearch故障转移"""
        # 切换到备用集群
        backup_cluster = self.backup_components['elasticsearch']
        await self.switch_elasticsearch_cluster(backup_cluster)

        # 重新索引数据
        await self.reindex_from_backup()
```

## 部署和扩展

### 1. **容器化部署**

#### Docker配置
```dockerfile
# Dockerfile for Audit Service
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes部署配置
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audit-service
  labels:
    app: audit-service
spec:
  replicas: 6
  selector:
    matchLabels:
      app: audit-service
  template:
    metadata:
      labels:
        app: audit-service
    spec:
      containers:
      - name: audit-service
        image: audit-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: KAFKA_BROKERS
          value: "kafka-1:9092,kafka-2:9092,kafka-3:9092"
        - name: ELASTICSEARCH_URL
          value: "http://elasticsearch:9200"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
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
  name: audit-service
spec:
  selector:
    app: audit-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2. **自动扩缩容**

#### HPA配置
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: audit-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: audit-service
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 3. **灾难恢复**

#### 数据备份策略
```python
class BackupManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.backup_schedule = {
            'full_backup': '0 2 * * 0',    # 每周日凌晨2点
            'incremental_backup': '0 */6 * * *'  # 每6小时
        }

    async def create_full_backup(self):
        """创建全量备份"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"audit_backup_full_{timestamp}"

        # 备份Elasticsearch数据
        es_snapshot = await self.create_elasticsearch_snapshot(backup_name)

        # 备份ClickHouse数据
        ch_backup = await self.create_clickhouse_backup(backup_name)

        # 上传到S3
        await self.upload_backup_to_s3(es_snapshot, ch_backup, backup_name)

        # 记录备份元数据
        await self.record_backup_metadata(backup_name, 'full')

    async def create_incremental_backup(self):
        """创建增量备份"""
        last_backup = await self.get_last_backup()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"audit_backup_incremental_{timestamp}"

        # 增量备份逻辑
        await self.create_incremental_es_backup(last_backup, backup_name)
        await self.create_incremental_ch_backup(last_backup, backup_name)

    async def restore_from_backup(self, backup_name: str):
        """从备份恢复"""
        backup_metadata = await self.get_backup_metadata(backup_name)

        if backup_metadata['type'] == 'full':
            await self.restore_full_backup(backup_name)
        else:
            await self.restore_incremental_backup(backup_name)
```

## 成本优化

### 1. **存储成本优化**

#### 生命周期管理
```python
class StorageLifecycleManager:
    def __init__(self):
        self.lifecycle_policies = {
            'hot_data': {'age_days': 30, 'storage_class': 'standard'},
            'warm_data': {'age_days': 365, 'storage_class': 'infrequent_access'},
            'cold_data': {'age_days': 2555, 'storage_class': 'glacier'},
            'delete_data': {'age_days': 3650, 'action': 'delete'}
        }

    async def apply_lifecycle_policies(self):
        """应用生命周期策略"""
        for index in await self.get_all_indices():
            age = await self.get_index_age(index)

            for policy_name, policy in self.lifecycle_policies.items():
                if age.days > policy['age_days']:
                    if policy['action'] == 'delete':
                        await self.delete_index(index)
                    else:
                        await self.change_storage_class(index, policy['storage_class'])
```

#### 数据压缩优化
```python
class CompressionOptimizer:
    def __init__(self):
        self.compression_algorithms = {
            'text_data': 'gzip',
            'json_data': 'lz4',
            'binary_data': 'zstd'
        }

    async def optimize_compression(self):
        """优化数据压缩"""
        # 分析数据类型
        data_types = await self.analyze_data_types()

        # 应用最佳压缩算法
        for data_type, algorithm in self.compression_algorithms.items():
            if data_type in data_types:
                await self.apply_compression(data_type, algorithm)
```

## 总结

大规模审计日志系统架构设计需要综合考虑：

1. **高性能**: 通过异步处理、批量操作、多级缓存保证性能
2. **高可用**: 通过多副本、故障转移、自动恢复保证可用性
3. **可扩展**: 通过水平扩展、分区策略、负载均衡支持增长
4. **成本优化**: 通过分层存储、生命周期管理、压缩优化控制成本
5. **运维友好**: 通过监控告警、自动化部署、灾难恢复简化运维

通过合理的架构设计和技术选型，可以构建出支持百万级用户的分布式审计日志系统。