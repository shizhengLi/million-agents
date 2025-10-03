# 服务发现系统面试题详解

## 概述

本文档包含了服务发现系统的核心面试题，涵盖基础概念、架构设计、实际应用和高级话题。这些问题反映了在百万级智能体社交平台项目中遇到的真实挑战和解决方案。

## 基础概念题

### 1. 什么是服务发现？为什么需要服务发现？

**回答思路**:
- 定义：服务发现是一种机制，允许服务自动注册其位置，并让其他服务能够找到它们
- 背景：微服务架构中服务的动态性
- 价值：解耦服务消费者和生产者，支持动态扩缩容

**参考答案**:
```
服务发现是分布式系统中的核心组件，主要解决两个问题：

1. 服务注册：服务启动时自动向注册中心注册自己的网络位置（IP、端口等）
2. 服务发现：服务消费者能够动态查找并获取服务提供者的位置信息

在现代微服务架构中，服务发现必不可少，因为：

a) 动态性：服务实例会频繁地启动、停止、迁移
b) 弹性伸缩：根据负载自动扩缩容实例数量
c) 故障转移：当某个实例故障时，需要快速发现并切换到健康实例
d) 环境复杂性：跨越多个数据中心、云环境的部署

没有服务发现，服务间的调用就会变成硬编码的配置，无法适应现代动态的分布式环境。
```

**代码示例**:
```python
class ServiceRegistry:
    def __init__(self):
        self.services = {}  # service_name -> [instances]
        self.health_checker = HealthChecker()

    def register_service(self, service_name: str, instance_info: dict):
        """服务注册"""
        if service_name not in self.services:
            self.services[service_name] = []

        self.services[service_name].append({
            "id": instance_info["id"],
            "address": instance_info["address"],
            "port": instance_info["port"],
            "metadata": instance_info.get("metadata", {}),
            "registered_at": time.time(),
            "status": "healthy"
        })

        # 启动健康检查
        self.health_checker.start_health_check(instance_info)

    def discover_service(self, service_name: str) -> list:
        """服务发现"""
        if service_name not in self.services:
            return []

        # 只返回健康的实例
        healthy_instances = [
            instance for instance in self.services[service_name]
            if instance["status"] == "healthy"
        ]

        return healthy_instances
```

### 2. 服务发现有哪些实现方式？各有什么优缺点？

**回答思路**:
- 分类：客户端发现 vs 服务端发现
- 具体实现：ZooKeeper、Eureka、Consul、etcd等
- 对比分析：一致性、可用性、性能等维度

**参考答案**:
```
服务发现主要有两种实现模式：

1. 客户端发现模式
   - 工作原理：客户端直接从注册中心查询服务实例，然后直接调用
   - 优点：减少网络跳转，性能更好；客户端可以做负载均衡
   - 缺点：客户端需要集成发现逻辑，增加复杂度；多语言支持困难

2. 服务端发现模式
   - 工作原理：客户端将请求发送到负载均衡器或API网关，由后者路由到具体实例
   - 优点：客户端逻辑简单，统一的流量控制，易于监控
   - 缺点：增加网络延迟，可能成为单点故障

具体实现方案对比：

ZooKeeper:
- 优点：强一致性，Watch机制，成熟稳定
- 缺点：复杂度高，性能相对较差，Java生态为主

Eureka:
- 优点：简单易用，AP系统保证可用性，自我保护机制
- 缺点：最终一致性，不再积极维护

Consul:
- 优点：功能丰富（服务发现+健康检查+KV存储），多数据中心支持
- 缺点：Gossip协议有一定延迟，Raft算法复杂

etcd:
- 优点：强一致性，高性能，云原生生态好
- 缺点：相对年轻，功能不如Consul丰富

Nacos:
- 优点：功能完整，支持多种发现模式，与Spring生态集成好
- 缺点：阿里巴巴项目，社区生态相对局限
```

## 架构设计题

### 3. 设计一个高可用的服务发现系统

**回答思路**:
- 整体架构：多层级、多区域部署
- 核心组件：注册中心、健康检查、配置管理
- 高可用策略：集群化、数据复制、故障转移
- 性能优化：缓存、批量操作、异步处理

**参考答案**:
```
设计一个高可用的服务发现系统需要考虑以下架构：

1. 整体架构
   ├─ 客户端层 (SDK/Agent)
   ├─ 接入层 (API Gateway/Load Balancer)
   ├─ 服务层 (Registry Cluster)
   ├─ 存储层 (Distributed Database)
   └─ 监控层 (Metrics & Alerts)

2. 核心组件设计

a) 服务注册中心集群
   - 使用Raft算法保证一致性
   - 多节点部署，至少3个节点
   - 跨数据中心部署

b) 健康检查系统
   - 主动检查：定期发送健康检查请求
   - 被动检查：基于心跳机制
   - 多层次检查：网络层、应用层、业务层

c) 配置管理中心
   - 集中管理服务配置
   - 支持动态配置更新
   - 配置版本控制和回滚

3. 高可用策略

a) 数据复制
   - 同步复制：关键数据强一致性
   - 异步复制：非关键数据最终一致性
   - 多区域复制：灾难恢复

b) 故障检测和恢复
   - 快速故障检测（秒级）
   - 自动故障转移
   - 节点自动恢复

4. 性能优化

a) 缓存策略
   - 客户端缓存：减少网络请求
   - 服务端缓存：提高查询性能
   - 多级缓存：内存+磁盘

b) 批量操作
   - 批量注册/注销
   - 批量健康检查
   - 批量配置更新
```

**代码示例**:
```python
class HighAvailabilityServiceRegistry:
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.raft_consensus = RaftConsensus(node_id, cluster_nodes)
        self.data_store = DistributedDataStore()
        self.health_checker = DistributedHealthChecker()
        self.cache_manager = CacheManager()

    async def register_service(self, service_info: dict) -> bool:
        """高可用服务注册"""
        try:
            # 1. 本地验证和预处理
            validated_info = self.validate_service_info(service_info)

            # 2. 通过Raft共识写入
            success = await self.raft_consensus.propose(
                operation="register",
                data=validated_info
            )

            if success:
                # 3. 更新本地数据
                await self.data_store.set(
                    key=f"service:{validated_info['service_name']}:{validated_info['instance_id']}",
                    value=validated_info
                )

                # 4. 清除相关缓存
                await self.cache_manager.invalidate_pattern(
                    f"service:{validated_info['service_name']}:*"
                )

                # 5. 启动健康检查
                await self.health_checker.add_service_instance(validated_info)

            return success

        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            return False

    async def discover_service(self, service_name: str) -> List[dict]:
        """高可用服务发现"""
        try:
            # 1. 尝试从缓存读取
            cache_key = f"service:{service_name}:instances"
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # 2. 从本地存储读取
            instances = await self.data_store.get_instances_by_service(service_name)

            # 3. 过滤健康实例
            healthy_instances = []
            for instance in instances:
                if await self.health_checker.is_healthy(instance["instance_id"]):
                    healthy_instances.append(instance)

            # 4. 缓存结果
            await self.cache_manager.set(cache_key, healthy_instances, ttl=30)

            return healthy_instances

        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            # 降级处理：返回缓存数据（即使过期）
            return await self.cache_manager.get(cache_key, allow_stale=True)

    async def handle_node_failure(self, failed_node_id: str):
        """处理节点故障"""
        logger.warning(f"Handling node failure: {failed_node_id}")

        # 1. 更新集群状态
        await self.raft_consensus.handle_node_failure(failed_node_id)

        # 2. 重新分配健康检查任务
        await self.health_checker.redistribute_checks(failed_node_id)

        # 3. 触发数据同步
        await self.data_store.sync_with_cluster()

class RaftConsensus:
    """Raft共识算法实现"""
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.current_term = 0
        self.state = "follower"  # follower, candidate, leader
        self.log = []
        self.commit_index = 0

    async def propose(self, operation: str, data: dict) -> bool:
        """提议操作"""
        if self.state != "leader":
            # 转发给leader
            leader = await self.get_current_leader()
            if leader:
                return await self.forward_to_leader(leader, operation, data)
            return False

        # Leader处理提议
        log_entry = {
            "term": self.current_term,
            "index": len(self.log) + 1,
            "operation": operation,
            "data": data
        }

        # 复制到多数节点
        success_count = await self.replicate_to_majority(log_entry)
        return success_count >= len(self.cluster_nodes) // 2 + 1

    async def replicate_to_majority(self, log_entry: dict) -> int:
        """复制到多数节点"""
        tasks = []
        for node in self.cluster_nodes:
            if node != self.node_id:
                task = asyncio.create_task(
                    self.send_log_entry(node, log_entry)
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = 1  # 自己算一个

        for result in results:
            if isinstance(result, bool) and result:
                success_count += 1

        return success_count
```

### 4. 如何实现服务发现的健康检查？

**回答思路**:
- 健康检查类型：主动检查、被动检查
- 检查层次：网络、应用、业务
- 检查策略：频率、超时、重试
- 故障处理：阈值判断、自动恢复

**参考答案**:
```
健康检查是服务发现系统的核心功能，确保只返回健康的服务实例：

1. 健康检查类型

a) 主动检查（Pull模式）
   - 定期向服务实例发送健康检查请求
   - 适用于HTTP、TCP、gRPC等协议
   - 可以检查应用状态和业务状态

b) 被动检查（Push模式）
   - 服务实例定期发送心跳到注册中心
   - 减少网络开销
   - 适用于大规模服务集群

2. 检查层次

a) 网络层检查
   - TCP连接检查
   - ICMP ping检查
   - 检查网络可达性

b) 应用层检查
   - HTTP健康检查接口
   - 自定义健康检查协议
   - 检查应用运行状态

c) 业务层检查
   - 数据库连接检查
   - 外部依赖检查
   - 业务指标检查

3. 检查策略

a) 检查频率
   - 关键服务：5-10秒
   - 普通服务：30-60秒
   - 批量检查：减少网络开销

b) 超时设置
   - 网络检查：2-3秒
   - 应用检查：5-10秒
   - 业务检查：10-30秒

c) 重试策略
   - 指数退避重试
   - 最大重试次数限制
   - 快速失败机制

4. 故障处理

a) 健康状态判断
   - 连续失败次数阈值
   - 成功率阈值
   - 响应时间阈值

b) 状态转换
   - 健康→可疑→不健康→恢复
   - 状态转换延迟（避免抖动）
   - 状态通知机制

c) 自动恢复
   - 检测到恢复后自动标记为健康
   - 重新加入负载均衡
   - 通知相关服务
```

**代码示例**:
```python
class HealthChecker:
    def __init__(self):
        self.check_interval = 30  # 默认30秒检查一次
        self.timeout = 5          # 默认5秒超时
        self.retry_count = 3      # 默认重试3次
        self.failure_threshold = 3  # 连续失败3次标记为不健康
        self.recovery_threshold = 2  # 连续成功2次标记为健康
        self.service_status = {}   # service_id -> status info
        self.check_tasks = {}      # service_id -> task

    async def add_service_instance(self, service_info: dict):
        """添加服务实例进行健康检查"""
        service_id = service_info["instance_id"]

        self.service_status[service_id] = {
            "service_info": service_info,
            "status": "healthy",  # healthy, suspicious, unhealthy
            "consecutive_failures": 0,
            "consecutive_successes": 0,
            "last_check_time": None,
            "last_success_time": None,
            "check_type": service_info.get("health_check_type", "http"),
            "check_config": service_info.get("health_check_config", {})
        }

        # 启动健康检查任务
        if service_id not in self.check_tasks:
            task = asyncio.create_task(self._health_check_loop(service_id))
            self.check_tasks[service_id] = task

    async def _health_check_loop(self, service_id: str):
        """健康检查循环"""
        while service_id in self.service_status:
            try:
                status_info = self.service_status[service_id]
                check_config = status_info["check_config"]

                # 执行健康检查
                is_healthy = await self._perform_health_check(
                    service_id,
                    status_info["check_type"],
                    check_config
                )

                # 更新状态
                await self._update_health_status(service_id, is_healthy)

                # 等待下次检查
                interval = check_config.get("interval", self.check_interval)
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {service_id}: {e}")
                await asyncio.sleep(5)

    async def _perform_health_check(self, service_id: str, check_type: str,
                                  check_config: dict) -> bool:
        """执行具体的健康检查"""
        service_info = self.service_status[service_id]["service_info"]
        address = service_info["address"]
        port = service_info["port"]
        timeout = check_config.get("timeout", self.timeout)

        if check_type == "http":
            return await self._http_health_check(
                address, port, check_config, timeout
            )
        elif check_type == "tcp":
            return await self._tcp_health_check(
                address, port, timeout
            )
        elif check_type == "grpc":
            return await self._grpc_health_check(
                address, port, check_config, timeout
            )
        elif check_type == "custom":
            return await self._custom_health_check(
                service_id, check_config, timeout
            )
        else:
            logger.warning(f"Unknown health check type: {check_type}")
            return False

    async def _http_health_check(self, address: str, port: int,
                               check_config: dict, timeout: int) -> bool:
        """HTTP健康检查"""
        import aiohttp

        path = check_config.get("path", "/health")
        expected_status = check_config.get("expected_status", 200)
        method = check_config.get("method", "GET")

        url = f"http://{address}:{port}{path}"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(method, url) as response:
                    return response.status == expected_status
        except Exception:
            return False

    async def _tcp_health_check(self, address: str, port: int, timeout: int) -> bool:
        """TCP健康检查"""
        try:
            future = asyncio.open_connection(address, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _update_health_status(self, service_id: str, is_healthy: bool):
        """更新健康状态"""
        status_info = self.service_status[service_id]
        current_time = time.time()

        status_info["last_check_time"] = current_time

        if is_healthy:
            status_info["consecutive_failures"] = 0
            status_info["consecutive_successes"] += 1
            status_info["last_success_time"] = current_time

            # 检查是否需要恢复状态
            if status_info["status"] == "unhealthy" and \
               status_info["consecutive_successes"] >= self.recovery_threshold:
                await self._mark_service_healthy(service_id)
            elif status_info["status"] == "suspicious":
                await self._mark_service_healthy(service_id)

        else:
            status_info["consecutive_failures"] += 1
            status_info["consecutive_successes"] = 0

            # 检查是否需要标记为不健康
            if status_info["consecutive_failures"] >= self.failure_threshold:
                await self._mark_service_unhealthy(service_id)
            elif status_info["status"] == "healthy":
                await self._mark_service_suspicious(service_id)

    async def _mark_service_healthy(self, service_id: str):
        """标记服务为健康状态"""
        old_status = self.service_status[service_id]["status"]
        self.service_status[service_id]["status"] = "healthy"

        if old_status != "healthy":
            logger.info(f"Service {service_id} recovered to healthy")
            await self._notify_status_change(service_id, "healthy")

    async def _mark_service_unhealthy(self, service_id: str):
        """标记服务为不健康状态"""
        old_status = self.service_status[service_id]["status"]
        self.service_status[service_id]["status"] = "unhealthy"

        if old_status != "unhealthy":
            logger.warning(f"Service {service_id} marked as unhealthy")
            await self._notify_status_change(service_id, "unhealthy")

    async def _notify_status_change(self, service_id: str, status: str):
        """通知服务状态变化"""
        # 发送状态变化事件
        event = {
            "type": "health_status_changed",
            "service_id": service_id,
            "status": status,
            "timestamp": time.time()
        }

        # 发布到消息队列
        await self.publish_event(event)

    def is_healthy(self, service_id: str) -> bool:
        """检查服务是否健康"""
        if service_id not in self.service_status:
            return False

        return self.service_status[service_id]["status"] == "healthy"
```

## 实际应用题

### 5. 在百万级智能体社交平台中，你是如何设计和实现服务发现的？

**回答思路**:
- 业务背景：智能体服务的特殊性
- 技术挑战：大规模、高频更新、多地域
- 解决方案：分层架构、优化策略
- 实际效果：性能指标、业务收益

**参考答案**:
```
在百万级智能体社交平台项目中，我们面临的服务发现挑战包括：

1. 业务挑战
   - 智能体服务数量庞大（10万+实例）
   - 服务状态变化频繁（智能体的上下线）
   - 多地域部署要求
   - 实时性要求高

2. 技术挑战
   - 海量服务实例的注册和发现
   - 高频的心跳和健康检查
   - 网络分区和故障处理
   - 性能和延迟要求

我们的解决方案：

1. 整体架构设计
   ├─ 全球服务发现层（跨地域）
   ├─ 区域服务发现层（同地域）
   ├─ 本地缓存层（客户端）
   └─ 监控告警层

2. 核心技术实现

a) 分层服务发现
   - 全球层：Consul集群，管理跨地域服务
   - 区域层：自研服务发现，管理区域内服务
   - 本地层：客户端缓存，减少网络请求

b) 智能体服务优化
   - 批量注册：智能体批量注册，减少网络开销
   - 增量更新：只同步变化的服务状态
   - 预测性下线：基于智能体行为预测下线

c) 性能优化
   - 多级缓存：内存缓存 + 本地文件缓存
   - 压缩传输：gzip压缩服务数据
   - 连接池：复用网络连接

3. 实际效果
   - 服务注册延迟：<100ms
   - 服务发现延迟：<50ms
   - 健康检查延迟：<5秒
   - 系统可用性：99.99%

4. 关键代码实现
```

**代码示例**:
```python
class IntelligentAgentServiceRegistry:
    """智能体服务专用注册中心"""

    def __init__(self):
        self.global_registry = ConsulClient()
        self.local_registry = LocalRegistry()
        self.cache_manager = MultiLevelCache()
        self.agent_predictor = AgentBehaviorPredictor()

    async def register_agent_batch(self, agents: List[dict]) -> dict:
        """批量注册智能体"""
        try:
            # 1. 预处理和验证
            validated_agents = []
            for agent in agents:
                validated_agent = self.validate_agent_info(agent)
                if validated_agent:
                    validated_agents.append(validated_agent)

            if not validated_agents:
                return {"success": False, "message": "No valid agents"}

            # 2. 本地注册
            local_results = []
            for agent in validated_agents:
                result = await self.local_registry.register(agent)
                local_results.append(result)

            # 3. 批量同步到全局注册中心
            global_success = await self._batch_sync_to_global(validated_agents)

            # 4. 更新缓存
            await self.cache_manager.invalidate_batch([
                f"agent:{agent['agent_id']}:*" for agent in validated_agents
            ])

            # 5. 启动行为预测
            for agent in validated_agents:
                await self.agent_predictor.start_monitoring(agent)

            return {
                "success": True,
                "registered_count": len(validated_agents),
                "local_success": sum(local_results),
                "global_success": global_success
            }

        except Exception as e:
            logger.error(f"Batch agent registration failed: {e}")
            return {"success": False, "message": str(e)}

    async def discover_agents(self, criteria: dict) -> List[dict]:
        """智能发现智能体"""
        try:
            cache_key = f"agents:{hash_dict(criteria)}"

            # 1. 尝试从缓存获取
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # 2. 从本地注册中心查询
            local_agents = await self.local_registry.discover(criteria)

            # 3. 过滤和排序
            filtered_agents = self._filter_agents(local_agents, criteria)
            sorted_agents = self._sort_agents(filtered_agents, criteria)

            # 4. 缓存结果
            await self.cache_manager.set(cache_key, sorted_agents, ttl=60)

            return sorted_agents

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return []

    async def predictive_deregistration(self, agent_id: str):
        """预测性注销智能体"""
        try:
            # 1. 预测智能体行为
            prediction = await self.agent_predictor.predict_offline_time(agent_id)

            if prediction["will_offline_soon"]:
                # 2. 优雅下线
                await self._graceful_shutdown(agent_id, prediction["estimated_offline_time"])

                # 3. 提前从负载均衡中移除
                await self._remove_from_load_balancer(agent_id)

                return {"success": True, "prediction": prediction}
            else:
                return {"success": False, "message": "Agent not predicted to go offline"}

        except Exception as e:
            logger.error(f"Predictive deregistration failed: {e}")
            return {"success": False, "message": str(e)}

    def _filter_agents(self, agents: List[dict], criteria: dict) -> List[dict]:
        """根据条件过滤智能体"""
        filtered = agents

        # 按能力过滤
        if "capabilities" in criteria:
            required_capabilities = set(criteria["capabilities"])
            filtered = [
                agent for agent in filtered
                if set(agent.get("capabilities", [])) & required_capabilities
            ]

        # 按状态过滤
        if "status" in criteria:
            filtered = [
                agent for agent in filtered
                if agent.get("status") == criteria["status"]
            ]

        # 按地域过滤
        if "region" in criteria:
            filtered = [
                agent for agent in filtered
                if agent.get("region") == criteria["region"]
            ]

        return filtered

    def _sort_agents(self, agents: List[dict], criteria: dict) -> List[dict]:
        """根据条件排序智能体"""
        if "sort_by" not in criteria:
            return agents

        sort_by = criteria["sort_by"]
        reverse = criteria.get("reverse", False)

        if sort_by == "performance":
            return sorted(agents,
                         key=lambda x: x.get("performance_score", 0),
                         reverse=reverse)
        elif sort_by == "availability":
            return sorted(agents,
                         key=lambda x: x.get("availability_rate", 0),
                         reverse=reverse)
        elif sort_by == "cost":
            return sorted(agents,
                         key=lambda x: x.get("cost_per_request", float('inf')),
                         reverse=reverse)
        else:
            return agents

class AgentBehaviorPredictor:
    """智能体行为预测器"""

    def __init__(self):
        self.behavior_models = {}
        self.historical_data = {}

    async def start_monitoring(self, agent_info: dict):
        """开始监控智能体行为"""
        agent_id = agent_info["agent_id"]

        # 初始化行为模型
        self.behavior_models[agent_id] = BehaviorModel(agent_info)

        # 启动数据收集
        asyncio.create_task(self._collect_behavior_data(agent_id))

    async def predict_offline_time(self, agent_id: str) -> dict:
        """预测智能体下线时间"""
        if agent_id not in self.behavior_models:
            return {"will_offline_soon": False}

        model = self.behavior_models[agent_id]
        historical_data = self.historical_data.get(agent_id, [])

        if len(historical_data) < 10:  # 数据不足
            return {"will_offline_soon": False}

        # 使用机器学习模型预测
        prediction = await model.predict_next_offline(historical_data)

        return prediction

    async def _collect_behavior_data(self, agent_id: str):
        """收集智能体行为数据"""
        while agent_id in self.behavior_models:
            try:
                # 收集当前行为数据
                current_data = await self._get_agent_behavior_data(agent_id)

                # 存储历史数据
                if agent_id not in self.historical_data:
                    self.historical_data[agent_id] = []

                self.historical_data[agent_id].append(current_data)

                # 保持数据量在合理范围
                if len(self.historical_data[agent_id]) > 1000:
                    self.historical_data[agent_id] = self.historical_data[agent_id][-1000:]

                # 更新行为模型
                await self.behavior_models[agent_id].update(current_data)

                await asyncio.sleep(60)  # 每分钟收集一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Behavior data collection error for {agent_id}: {e}")
                await asyncio.sleep(10)
```

## 高级话题题

### 6. 如何处理服务发现中的网络分区问题？

**回答思路**:
- 网络分区的定义和影响
- 分区检测机制
- 分区期间的策略选择
- 分区恢复后的数据同步

**参考答案**:
```
网络分区是分布式系统中不可避免的问题，在服务发现中需要特别处理：

1. 网络分区的影响
   - 节点间无法通信
   - 数据不一致
   - 服务发现失败
   - 重复注册问题

2. 分区检测机制
   a) 心跳超时检测
   b) Gossip协议检测
   c) 法定人数投票检测
   d) 外部监控系统检测

3. 分区处理策略
   a) 只读模式
   - 停止接受写操作
   - 继续提供读服务
   - 保证数据一致性

   b) 分区可用模式
   - 分区内继续正常服务
   - 允许数据不一致
   - 分区恢复后同步

   c) 混合模式
   - 核心服务只读
   - 非核心服务可用
   - 根据重要性区分处理

4. 分区恢复策略
   a) 数据同步
   - 增量同步
   - 全量同步
   - 冲突解决

   b) 服务重新注册
   - 自动重新注册
   - 状态验证
   - 负载均衡更新

5. 实际实现方案
```

**代码示例**:
```python
class PartitionAwareServiceRegistry:
    """支持网络分区的服务注册中心"""

    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.partition_detector = PartitionDetector()
        self.partition_handler = PartitionHandler()
        self.data_sync = DataSynchronization()

    async def handle_partition_detection(self):
        """处理分区检测"""
        asyncio.create_task(self._partition_detection_loop())

    async def _partition_detection_loop(self):
        """分区检测循环"""
        while True:
            try:
                partition_info = await self.partition_detector.detect_partition()

                if partition_info.is_partitioned:
                    await self._handle_network_partition(partition_info)
                else:
                    await self._handle_partition_recovery()

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Partition detection error: {e}")
                await asyncio.sleep(1)

    async def _handle_network_partition(self, partition_info):
        """处理网络分区"""
        logger.warning(f"Network partition detected: {partition_info}")

        # 1. 判断分区类型
        partition_type = self._determine_partition_type(partition_info)

        # 2. 根据分区类型选择策略
        if partition_type == "minority_partition":
            # 少数派分区，切换到只读模式
            await self._switch_to_read_only_mode()
        elif partition_type == "majority_partition":
            # 多数派分区，继续提供服务
            await self._continue_normal_service()
        elif partition_type == "split_brain":
            # 脑裂，需要特殊处理
            await self._handle_split_brain()

    def _determine_partition_type(self, partition_info) -> str:
        """判断分区类型"""
        connected_nodes = partition_info.connected_nodes
        total_nodes = len(self.cluster_nodes)
        connected_count = len(connected_nodes)

        if connected_count < total_nodes // 2 + 1:
            return "minority_partition"
        elif connected_count == total_nodes:
            return "no_partition"
        else:
            # 检查是否存在其他多数派分区
            if self._detect_split_brain(partition_info):
                return "split_brain"
            else:
                return "majority_partition"

    async def _switch_to_read_only_mode(self):
        """切换到只读模式"""
        logger.info("Switching to read-only mode due to network partition")

        # 1. 停止接受写操作
        self.write_enabled = False

        # 2. 通知客户端
        await self._notify_clients_mode_change("read_only")

        # 3. 继续提供读服务
        await self._enable_read_only_service()

    async def _handle_partition_recovery(self):
        """处理分区恢复"""
        logger.info("Network partition recovered, starting reconciliation")

        # 1. 重新连接到集群
        await self._reconnect_to_cluster()

        # 2. 数据同步
        await self.data_sync.sync_with_cluster()

        # 3. 恢复正常服务
        await self._restore_normal_service()

    async def _restore_normal_service(self):
        """恢复正常服务"""
        logger.info("Restoring normal service mode")

        # 1. 启用写操作
        self.write_enabled = True

        # 2. 重新启动健康检查
        await self._restart_health_checks()

        # 3. 通知客户端
        await self._notify_clients_mode_change("normal")

class PartitionDetector:
    """网络分区检测器"""

    def __init__(self):
        self.heartbeat_interval = 5
        self.partition_threshold = 15  # 15秒无心跳认为分区

    async def detect_partition(self) -> PartitionInfo:
        """检测网络分区"""
        connected_nodes = set()
        failed_nodes = set()

        for node_id in self.cluster_nodes:
            if node_id == self.node_id:
                connected_nodes.add(node_id)
                continue

            try:
                # 发送心跳检测
                if await self._send_heartbeat(node_id):
                    connected_nodes.add(node_id)
                else:
                    failed_nodes.add(node_id)
            except Exception:
                failed_nodes.add(node_id)

        # 判断是否分区
        is_partitioned = len(failed_nodes) > 0

        return PartitionInfo(
            is_partitioned=is_partitioned,
            connected_nodes=connected_nodes,
            failed_nodes=failed_nodes,
            detection_time=time.time()
        )

    async def _send_heartbeat(self, node_id: str) -> bool:
        """发送心跳"""
        try:
            # 实现心跳逻辑
            return True
        except Exception:
            return False

@dataclass
class PartitionInfo:
    is_partitioned: bool
    connected_nodes: Set[str]
    failed_nodes: Set[str]
    detection_time: float
```

## 总结

服务发现系统的面试题主要考察：

1. **基础概念理解**: 服务发现的原理和价值
2. **架构设计能力**: 高可用、高性能的系统设计
3. **实际实现经验**: 健康检查、故障处理等具体实现
4. **问题解决能力**: 网络分区、大规模场景等挑战的处理
5. **业务场景应用**: 结合具体业务场景的解决方案

在面试中，应该结合具体的项目经验，说明设计决策的思考过程和实际效果，展示对分布式系统原理的深入理解。

---

**相关阅读**:
- [服务发现架构设计](../service-discovery/architecture.md)
- [分布式系统核心概念](../knowledge-base/core-concepts.md)
- [负载均衡器面试题](./load-balancer.md)