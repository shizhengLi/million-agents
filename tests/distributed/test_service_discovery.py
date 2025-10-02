"""
服务发现和健康检查测试
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from src.distributed.service_discovery import (
    ServiceDiscovery, ServiceRegistry, ServiceInstance,
    HealthChecker, HealthStatus, ServiceType
)


class TestServiceInstance:
    """测试服务实例"""

    def test_service_instance_creation(self):
        """测试服务实例创建"""
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001,
            metadata={"region": "us-west-1"}
        )

        assert instance.id == "service_1"
        assert instance.service_type == ServiceType.WORKER_NODE
        assert instance.address == "localhost:8001"
        assert instance.port == 8001
        assert instance.metadata == {"region": "us-west-1"}
        assert instance.status == HealthStatus.UNKNOWN
        assert instance.registered_at > 0
        assert instance.last_heartbeat is None
        assert instance.health_check_count == 0

    def test_service_instance_update_heartbeat(self):
        """测试更新心跳"""
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        initial_time = instance.registered_at
        time.sleep(0.01)
        instance.update_heartbeat()

        assert instance.last_heartbeat is not None
        assert instance.last_heartbeat > initial_time

    def test_service_instance_mark_healthy(self):
        """测试标记服务健康"""
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        instance.mark_healthy()
        assert instance.status == HealthStatus.HEALTHY
        assert instance.health_check_count > 0

    def test_service_instance_mark_unhealthy(self):
        """测试标记服务不健康"""
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        instance.mark_unhealthy("Connection timeout")
        assert instance.status == HealthStatus.UNHEALTHY
        assert instance.error_message == "Connection timeout"
        assert instance.health_check_count > 0

    def test_service_instance_is_expired(self):
        """测试服务实例是否过期"""
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        # 刚注册的实例不应该过期
        assert not instance.is_expired(ttl_seconds=60)

        # 模拟过期
        instance.last_heartbeat = time.time() - 120
        assert instance.is_expired(ttl_seconds=60)

    def test_service_instance_get_endpoint(self):
        """测试获取服务端点"""
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost",
            port=8001
        )

        assert instance.get_endpoint() == "localhost:8001"
        assert instance.get_endpoint(scheme="http") == "http://localhost:8001"


class TestServiceRegistry:
    """测试服务注册表"""

    def test_service_registry_initialization(self):
        """测试服务注册表初始化"""
        registry = ServiceRegistry()

        assert len(registry.services) == 0
        assert registry.default_ttl == 60
        assert registry.cleanup_interval == 30

    def test_register_service(self):
        """测试注册服务"""
        registry = ServiceRegistry()
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        result = registry.register_service(instance)
        assert result is True
        assert "service_1" in registry.services
        assert registry.services["service_1"] == instance

    def test_register_duplicate_service(self):
        """测试注册重复服务"""
        registry = ServiceRegistry()
        instance1 = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        instance2 = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )

        registry.register_service(instance1)
        result = registry.register_service(instance2)

        # 应该更新现有服务
        assert result is True
        assert registry.services["service_1"].port == 8002

    def test_unregister_service(self):
        """测试注销服务"""
        registry = ServiceRegistry()
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        registry.register_service(instance)
        result = registry.unregister_service("service_1")

        assert result is True
        assert "service_1" not in registry.services

    def test_unregister_nonexistent_service(self):
        """测试注销不存在的服务"""
        registry = ServiceRegistry()
        result = registry.unregister_service("nonexistent")
        assert result is False

    def test_get_service(self):
        """测试获取服务"""
        registry = ServiceRegistry()
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        registry.register_service(instance)
        retrieved = registry.get_service("service_1")

        assert retrieved == instance

    def test_get_nonexistent_service(self):
        """测试获取不存在的服务"""
        registry = ServiceRegistry()
        retrieved = registry.get_service("nonexistent")
        assert retrieved is None

    def test_get_services_by_type(self):
        """测试按类型获取服务"""
        registry = ServiceRegistry()

        worker1 = ServiceInstance(
            id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        worker2 = ServiceInstance(
            id="worker_2",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )
        coordinator = ServiceInstance(
            id="coordinator_1",
            service_type=ServiceType.COORDINATOR,
            address="localhost:9001",
            port=9001
        )

        registry.register_service(worker1)
        registry.register_service(worker2)
        registry.register_service(coordinator)

        workers = registry.get_services_by_type(ServiceType.WORKER_NODE)
        coordinators = registry.get_services_by_type(ServiceType.COORDINATOR)

        assert len(workers) == 2
        assert len(coordinators) == 1
        assert all(s.service_type == ServiceType.WORKER_NODE for s in workers)

    def test_get_healthy_services(self):
        """测试获取健康服务"""
        registry = ServiceRegistry()

        healthy = ServiceInstance(
            id="healthy",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        unhealthy = ServiceInstance(
            id="unhealthy",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )
        unknown = ServiceInstance(
            id="unknown",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8003",
            port=8003
        )

        healthy.mark_healthy()
        unhealthy.mark_unhealthy("Error")

        registry.register_service(healthy)
        registry.register_service(unhealthy)
        registry.register_service(unknown)

        healthy_services = registry.get_healthy_services()

        assert len(healthy_services) == 1
        assert healthy_services[0].id == "healthy"

    def test_update_service_heartbeat(self):
        """测试更新服务心跳"""
        registry = ServiceRegistry()
        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        registry.register_service(instance)
        # 手动设置初始心跳
        instance.last_heartbeat = time.time()
        initial_heartbeat = instance.last_heartbeat

        time.sleep(0.01)
        result = registry.update_service_heartbeat("service_1")

        assert result is True
        assert instance.last_heartbeat > initial_heartbeat

    def test_update_nonexistent_heartbeat(self):
        """测试更新不存在服务的心跳"""
        registry = ServiceRegistry()
        result = registry.update_service_heartbeat("nonexistent")
        assert result is False

    def test_cleanup_expired_services(self):
        """测试清理过期服务"""
        registry = ServiceRegistry()

        fresh = ServiceInstance(
            id="fresh",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        expired = ServiceInstance(
            id="expired",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )

        # 模拟过期
        expired.last_heartbeat = time.time() - 120
        expired.registered_at = time.time() - 120

        registry.register_service(fresh)
        registry.register_service(expired)

        cleaned_count = registry.cleanup_expired_services(ttl_seconds=60)

        assert cleaned_count == 1
        assert len(registry.services) == 1
        assert "fresh" in registry.services
        assert "expired" not in registry.services

    def test_get_service_statistics(self):
        """测试获取服务统计信息"""
        registry = ServiceRegistry()

        worker1 = ServiceInstance(
            id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        worker2 = ServiceInstance(
            id="worker_2",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )
        coordinator = ServiceInstance(
            id="coordinator_1",
            service_type=ServiceType.COORDINATOR,
            address="localhost:9001",
            port=9001
        )

        worker1.mark_healthy()
        worker2.mark_unhealthy("Error")
        coordinator.mark_healthy()

        registry.register_service(worker1)
        registry.register_service(worker2)
        registry.register_service(coordinator)

        stats = registry.get_statistics()

        assert stats['total_services'] == 3
        assert stats['healthy_services'] == 2
        assert stats['unhealthy_services'] == 1
        assert stats['services_by_type'][ServiceType.WORKER_NODE.value] == 2
        assert stats['services_by_type'][ServiceType.COORDINATOR.value] == 1


class TestHealthChecker:
    """测试健康检查器"""

    def test_health_checker_initialization(self):
        """测试健康检查器初始化"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry)

        assert checker.registry == registry
        assert checker.check_interval == 30
        assert checker.timeout == 10
        assert checker.is_running is False

    def test_health_checker_custom_config(self):
        """测试健康检查器自定义配置"""
        registry = ServiceRegistry()
        checker = HealthChecker(
            registry=registry,
            check_interval=60,
            timeout=20
        )

        assert checker.check_interval == 60
        assert checker.timeout == 20

    @pytest.mark.asyncio
    async def test_start_stop_health_checking(self):
        """测试启动和停止健康检查"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry)

        await checker.start()
        assert checker.is_running is True

        await checker.stop()
        assert checker.is_running is False

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """测试启动已运行的健康检查器"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry)

        await checker.start()
        await checker.start()  # 应该不会出错

        await checker.stop()

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """测试停止未运行的健康检查器"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry)

        await checker.stop()  # 应该不会出错
        assert checker.is_running is False

    @pytest.mark.asyncio
    async def test_check_service_health_healthy(self):
        """测试检查健康服务"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry)

        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="httpbin.org",
            port=80
        )
        instance.mark_healthy()

        # Mock HTTP请求成功
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            await checker._check_service_health(instance)

            assert instance.status == HealthStatus.HEALTHY
            assert instance.health_check_count > 0

    @pytest.mark.asyncio
    async def test_check_service_health_unhealthy(self):
        """测试检查不健康服务"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry)

        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="nonexistent-host",
            port=8001
        )

        # Mock HTTP请求失败
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            await checker._check_service_health(instance)

            assert instance.status == HealthStatus.UNHEALTHY
            assert "Connection failed" in instance.error_message

    @pytest.mark.asyncio
    async def test_health_check_loop(self):
        """测试健康检查循环"""
        registry = ServiceRegistry()
        checker = HealthChecker(registry, check_interval=0.01)

        instance = ServiceInstance(
            id="service_1",
            service_type=ServiceType.WORKER_NODE,
            address="test-host",
            port=8001
        )

        registry.register_service(instance)

        # Mock健康检查
        with patch.object(checker, '_check_service_health', new_callable=AsyncMock) as mock_check:
            await checker.start()
            await asyncio.sleep(0.05)  # 等待几次检查
            await checker.stop()

            # 应该被检查了至少一次
            assert mock_check.call_count > 0


class TestServiceDiscovery:
    """测试服务发现"""

    def test_service_discovery_initialization(self):
        """测试服务发现初始化"""
        discovery = ServiceDiscovery()

        assert discovery.registry is not None
        assert discovery.health_checker is not None
        assert discovery.is_running is False

    @pytest.mark.asyncio
    async def test_start_stop_service_discovery(self):
        """测试启动和停止服务发现"""
        discovery = ServiceDiscovery()

        await discovery.start()
        assert discovery.is_running is True

        await discovery.stop()
        assert discovery.is_running is False

    def test_register_service_instance(self):
        """测试注册服务实例"""
        discovery = ServiceDiscovery()

        result = discovery.register_service(
            service_id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001,
            metadata={"region": "us-west-1"}
        )

        assert result is True
        service = discovery.get_service("worker_1")
        assert service is not None
        assert service.id == "worker_1"
        assert service.service_type == ServiceType.WORKER_NODE

    def test_unregister_service_instance(self):
        """测试注销服务实例"""
        discovery = ServiceDiscovery()

        discovery.register_service(
            service_id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        result = discovery.unregister_service("worker_1")
        assert result is True

        service = discovery.get_service("worker_1")
        assert service is None

    def test_discover_services_by_type(self):
        """测试按类型发现服务"""
        discovery = ServiceDiscovery()

        discovery.register_service(
            service_id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        discovery.register_service(
            service_id="worker_2",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )
        discovery.register_service(
            service_id="coordinator_1",
            service_type=ServiceType.COORDINATOR,
            address="localhost:9001",
            port=9001
        )

        workers = discovery.discover_services(ServiceType.WORKER_NODE)
        coordinators = discovery.discover_services(ServiceType.COORDINATOR)

        assert len(workers) == 2
        assert len(coordinators) == 1

    def test_discover_healthy_services_only(self):
        """测试只发现健康服务"""
        discovery = ServiceDiscovery()

        # 注册服务
        discovery.register_service(
            service_id="healthy_worker",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        discovery.register_service(
            service_id="unhealthy_worker",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002
        )

        # 手动设置健康状态
        healthy_service = discovery.get_service("healthy_worker")
        unhealthy_service = discovery.get_service("unhealthy_worker")

        healthy_service.mark_healthy()
        unhealthy_service.mark_unhealthy("Connection failed")

        # 只发现健康服务
        healthy_workers = discovery.discover_services(
            ServiceType.WORKER_NODE,
            healthy_only=True
        )

        assert len(healthy_workers) == 1
        assert healthy_workers[0].id == "healthy_worker"

    def test_get_load_balanced_service(self):
        """测试获取负载均衡的服务"""
        discovery = ServiceDiscovery()

        discovery.register_service(
            service_id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001,
            metadata={"weight": 1}
        )
        discovery.register_service(
            service_id="worker_2",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8002",
            port=8002,
            metadata={"weight": 2}
        )

        # 手动标记服务为健康
        worker1 = discovery.get_service("worker_1")
        worker2 = discovery.get_service("worker_2")
        worker1.mark_healthy()
        worker2.mark_healthy()

        service = discovery.get_load_balanced_service(ServiceType.WORKER_NODE)
        assert service is not None
        assert service.service_type == ServiceType.WORKER_NODE

    def test_get_load_balanced_service_none_available(self):
        """测试没有可用服务时的负载均衡"""
        discovery = ServiceDiscovery()

        service = discovery.get_load_balanced_service(ServiceType.WORKER_NODE)
        assert service is None

    def test_update_service_heartbeat(self):
        """测试更新服务心跳"""
        discovery = ServiceDiscovery()

        discovery.register_service(
            service_id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        result = discovery.update_heartbeat("worker_1")
        assert result is True

        service = discovery.get_service("worker_1")
        assert service.last_heartbeat is not None

    def test_get_discovery_statistics(self):
        """测试获取服务发现统计信息"""
        discovery = ServiceDiscovery()

        discovery.register_service(
            service_id="worker_1",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )
        discovery.register_service(
            service_id="coordinator_1",
            service_type=ServiceType.COORDINATOR,
            address="localhost:9001",
            port=9001
        )

        stats = discovery.get_statistics()

        assert 'registry' in stats
        assert 'health_checker' in stats
        assert stats['registry']['total_services'] == 2
        assert stats['health_checker']['is_running'] is False

    def test_service_type_validation(self):
        """测试服务类型验证"""
        # 测试所有有效的服务类型
        for service_type in ServiceType:
            assert isinstance(service_type.value, str)
            assert len(service_type.value) > 0

    def test_health_status_transitions(self):
        """测试健康状态转换"""
        instance = ServiceInstance(
            id="test_service",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        # 初始状态
        assert instance.status == HealthStatus.UNKNOWN

        # 转换到健康
        instance.mark_healthy()
        assert instance.status == HealthStatus.HEALTHY

        # 转换到不健康
        instance.mark_unhealthy("Test error")
        assert instance.status == HealthStatus.UNHEALTHY

        # 回到健康
        instance.mark_healthy()
        assert instance.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_service_discovery_full_lifecycle(self):
        """测试服务发现完整生命周期"""
        discovery = ServiceDiscovery()

        # 启动服务发现
        await discovery.start()

        # 注册服务
        service_id = discovery.register_service(
            service_id="test_worker",
            service_type=ServiceType.WORKER_NODE,
            address="localhost:8001",
            port=8001
        )

        assert service_id is True

        # 发现服务
        services = discovery.discover_services(ServiceType.WORKER_NODE)
        assert len(services) == 1

        # 更新心跳
        result = discovery.update_heartbeat("test_worker")
        assert result is True

        # 获取统计信息
        stats = discovery.get_statistics()
        assert stats['registry']['total_services'] == 1

        # 注销服务
        result = discovery.unregister_service("test_worker")
        assert result is True

        # 停止服务发现
        await discovery.stop()

        # 验证服务已移除
        services = discovery.discover_services(ServiceType.WORKER_NODE)
        assert len(services) == 0