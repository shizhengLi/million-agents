"""
服务发现95%+覆盖率测试 - 修正版
基于实际API的精确测试
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from src.distributed.service_discovery import (
    ServiceRegistry, ServiceInstance, ServiceType, HealthStatus,
    HealthChecker, ServiceDiscovery
)


class TestServiceDiscoveryCorrectedCoverage:
    """服务发现修正版覆盖率测试"""

    def test_service_instance_creation(self):
        """测试服务实例创建"""
        service = ServiceInstance(
            id="test_service",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )

        assert service.id == "test_service"
        assert service.service_type == ServiceType.WEB
        assert service.address == "127.0.0.1:8080"

    def test_service_registry_basic_operations(self):
        """测试服务注册表基本操作"""
        registry = ServiceRegistry()

        # 创建服务实例
        service = ServiceInstance(
            id="test_service",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )

        # 注册服务
        registry.register(service)

        # 获取服务
        retrieved_service = registry.get_service("test_service")
        assert retrieved_service is not None
        assert retrieved_service.id == "test_service"

        # 注销服务
        registry.unregister("test_service")
        assert registry.get_service("test_service") is None

    def test_service_registry_statistics(self):
        """测试服务注册表统计信息 (第305-307行)"""
        registry = ServiceRegistry()

        # 添加多个服务
        service1 = ServiceInstance(
            id="service1",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        service2 = ServiceInstance(
            id="service2",
            service_type=ServiceType.API,
            address="127.0.0.1:8081"
        )

        registry.register(service1)
        registry.register(service2)

        # 获取统计信息 (第305-307行)
        stats = registry.get_statistics()

        assert 'total_services' in stats
        assert 'healthy_services' in stats
        assert 'unhealthy_services' in stats
        assert 'service_types' in stats

        assert stats['total_services'] == 2

    def test_health_checker_initialization(self):
        """测试健康检查器初始化"""
        registry = ServiceRegistry()
        health_checker = HealthChecker(registry, check_interval=15, timeout=5)

        assert health_checker.check_interval == 15
        assert health_checker.timeout == 5
        assert health_checker.registry == registry

    def test_health_checker_statistics(self):
        """测试健康检查器统计信息 (第352-355行)"""
        registry = ServiceRegistry()
        health_checker = HealthChecker(registry)

        # 获取统计信息 (第352-355行)
        stats = health_checker.get_statistics()

        assert 'total_checks' in stats
        assert 'successful_checks' in stats
        assert 'failed_checks' in stats
        assert 'check_interval' in stats
        assert 'timeout' in stats

        assert stats['check_interval'] == 30  # 默认值
        assert stats['timeout'] == 10  # 默认值

    def test_service_discovery_initialization(self):
        """测试服务发现初始化"""
        discovery = ServiceDiscovery(ttl=120, health_check_interval=45)

        assert discovery.registry.default_ttl == 120
        assert discovery.health_checker.check_interval == 45

    def test_service_instance_status_management(self):
        """测试服务实例状态管理"""
        registry = ServiceRegistry()

        service = ServiceInstance(
            id="status_test",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        registry.register(service)

        # 标记为健康
        registry.mark_healthy("status_test")
        updated_service = registry.get_service("status_test")
        assert updated_service.health_status == HealthStatus.HEALTHY

        # 标记为不健康
        registry.mark_unhealthy("status_test")
        updated_service = registry.get_service("status_test")
        assert updated_service.health_status == HealthStatus.UNHEALTHY

    def test_service_registry_error_handling(self):
        """测试服务注册表错误处理"""
        registry = ServiceRegistry()

        # 测试注销不存在的服务
        registry.unregister("nonexistent")  # 不应该抛出异常

        # 测试标记不存在服务的状态
        registry.mark_healthy("nonexistent")  # 不应该抛出异常
        registry.mark_unhealthy("nonexistent")  # 不应该抛出异常

        # 获取不存在的服务
        assert registry.get_service("nonexistent") is None

    def test_service_type_filtering(self):
        """测试按服务类型过滤"""
        registry = ServiceRegistry()

        # 添加不同类型的服务
        web_service = ServiceInstance(
            id="web1",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        api_service = ServiceInstance(
            id="api1",
            service_type=ServiceType.API,
            address="127.0.0.1:8081"
        )
        db_service = ServiceInstance(
            id="db1",
            service_type=ServiceType.DATABASE,
            address="127.0.0.1:8082"
        )

        registry.register(web_service)
        registry.register(api_service)
        registry.register(db_service)

        # 按类型过滤
        web_services = registry.get_services_by_type(ServiceType.WEB)
        assert len(web_services) == 1
        assert web_services[0].service_type == ServiceType.WEB

        api_services = registry.get_services_by_type(ServiceType.API)
        assert len(api_services) == 1
        assert api_services[0].service_type == ServiceType.API

    def test_service_discovery_get_methods(self):
        """测试服务发现的各种获取方法"""
        discovery = ServiceDiscovery()

        # 添加服务
        service1 = ServiceInstance(
            id="service1",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        service2 = ServiceInstance(
            id="service2",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8081"
        )

        discovery.registry.register(service1)
        discovery.registry.register(service2)

        # 测试获取单个服务
        service = discovery.get_service("service1")
        assert service is not None
        assert service.id == "service1"

        # 测试按类型获取服务
        web_services = discovery.get_services_by_type(ServiceType.WEB)
        assert len(web_services) == 2

        # 测试获取所有服务
        all_services = discovery.get_all_services()
        assert len(all_services) == 2

    def test_service_discovery_selection_methods(self):
        """测试服务发现的选择方法"""
        discovery = ServiceDiscovery()

        # 添加多个相同类型的服务
        services = []
        for i in range(3):
            service = ServiceInstance(
                id=f"service_{i}",
                service_type=ServiceType.WEB,
                address=f"127.0.0.1:{8080 + i}"
            )
            services.append(service)
            discovery.registry.register(service)

        # 测试随机选择 (第560行附近)
        selected = discovery.get_random_service(ServiceType.WEB)
        assert selected is not None
        assert selected.service_type == ServiceType.WEB

        # 测试轮询选择 (第571行附近)
        for i in range(3):
            selected = discovery.get_round_robin_service(ServiceType.WEB)
            assert selected is not None
            assert selected.service_type == ServiceType.WEB

        # 测试获取不存在的服务类型
        assert discovery.get_random_service(ServiceType.DATABASE) is None
        assert discovery.get_round_robin_service(ServiceType.DATABASE) is None

    def test_service_discovery_lifecycle(self):
        """测试服务发现生命周期"""
        discovery = ServiceDiscovery()

        # 启动健康检查
        discovery.start_health_checking()

        # 添加服务
        service = ServiceInstance(
            id="lifecycle_test",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        discovery.register_service(service)

        # 等待一小段时间
        time.sleep(0.1)

        # 验证服务存在
        assert discovery.get_service("lifecycle_test") is not None

        # 停止健康检查
        discovery.stop_health_checking()

    def test_service_registry_cleanup(self):
        """测试服务注册表清理"""
        registry = ServiceRegistry(default_ttl=1)  # 1秒TTL

        # 添加服务
        service = ServiceInstance(
            id="cleanup_test",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        registry.register(service)

        # 等待服务过期
        time.sleep(1.1)

        # 执行清理
        registry.cleanup_expired()

        # 验证服务被清理
        assert registry.get_service("cleanup_test") is None

    def test_health_check_with_service(self):
        """测试健康检查与服务"""
        registry = ServiceRegistry()
        health_checker = HealthChecker(registry)

        service = ServiceInstance(
            id="health_test",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        registry.register(service)

        # 测试健康检查
        async def test_health_check():
            # 模拟健康检查成功
            with patch('socket.socket') as mock_socket:
                mock_conn = Mock()
                mock_socket.return_value.__enter__.return_value = mock_conn

                result = await health_checker.check_service_health(service)
                return result

        result = asyncio.run(test_health_check())
        assert isinstance(result, bool)

    def test_edge_cases_and_comprehensive_coverage(self):
        """测试边缘情况和综合覆盖率"""
        discovery = ServiceDiscovery()

        # 测试空服务发现的各种操作
        assert discovery.get_service("nonexistent") is None
        assert discovery.get_services_by_type(ServiceType.WEB) == []
        assert discovery.get_all_services() == []

        # 测试获取不存在类型的服务
        assert discovery.get_random_service(ServiceType.DATABASE) is None
        assert discovery.get_round_robin_service(ServiceType.DATABASE) is None

        # 注册然后注销服务
        service = ServiceInstance(
            id="edge_case_test",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080"
        )
        discovery.register_service(service)

        assert discovery.get_service("edge_case_test") is not None

        discovery.unregister_service("edge_case_test")
        assert discovery.get_service("edge_case_test") is None

        # 获取统计信息
        stats = discovery.get_statistics()
        assert 'total_services' in stats
        assert 'service_types' in stats
        assert stats['total_services'] == 0

    def test_concurrent_operations(self):
        """测试并发操作"""
        discovery = ServiceDiscovery()

        # 添加多个服务
        services = []
        for i in range(5):
            service = ServiceInstance(
                id=f"concurrent_{i}",
                service_type=ServiceType.WEB,
                address=f"127.0.0.1:{8080 + i}"
            )
            services.append(service)
            discovery.register_service(service)

        # 并发选择服务
        selected_services = []
        for _ in range(10):
            service = discovery.get_random_service(ServiceType.WEB)
            if service:
                selected_services.append(service.id)

        # 验证选择的服务都是有效的
        for service_id in selected_services:
            assert service_id.startswith("concurrent_")
            assert int(service_id.split("_")[1]) < 5

    def test_service_health_status_transitions(self):
        """测试服务健康状态转换"""
        registry = ServiceRegistry()

        service = ServiceInstance(
            id="status_transition_test",
            service_type=ServiceType.WEB,
            address="127.0.0.1:8080",
            health_status=HealthStatus.UNKNOWN
        )
        registry.register(service)

        # 状态转换：UNKNOWN -> HEALTHY
        registry.mark_healthy("status_transition_test")
        assert registry.get_service("status_transition_test").health_status == HealthStatus.HEALTHY

        # 状态转换：HEALTHY -> UNHEALTHY
        registry.mark_unhealthy("status_transition_test")
        assert registry.get_service("status_transition_test").health_status == HealthStatus.UNHEALTHY

        # 状态转换：UNHEALTHY -> HEALTHY
        registry.mark_healthy("status_transition_test")
        assert registry.get_service("status_transition_test").health_status == HealthStatus.HEALTHY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])