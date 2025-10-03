"""
服务发现95%+覆盖率测试
针对TDD的精确测试，确保覆盖所有未覆盖的行
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from src.distributed.service_discovery import ServiceRegistry, ServiceInstance


class TestServiceDiscoveryCoverage95Plus:
    """服务发现95%+覆盖率测试"""

    def test_service_instance_metadata_and_timeout(self):
        """测试服务实例元数据和超时配置 (第71-73行)"""
        service = ServiceInstance(
            id="test_service",
            name="test",
            host="127.0.0.1",
            port=8080,
            metadata={"version": "1.0", "region": "us-west"},
            timeout=30
        )

        # 测试详细信息包含超时配置 (第71-73行)
        details = service.get_details()
        assert 'timeout' in details
        assert details['timeout'] == 30
        assert details['metadata'] == {"version": "1.0", "region": "us-west"}

    def test_service_registry_detailed_statistics(self):
        """测试服务注册表的详细统计信息 (第305-307行)"""
        registry = ServiceRegistry()

        # 添加不同状态的服务
        service1 = ServiceInstance(id="healthy1", name="test", host="127.0.0.1", port=8080)
        service2 = ServiceInstance(id="healthy2", name="test", host="127.0.0.1", port=8081)
        service3 = ServiceInstance(id="unhealthy", name="test", host="127.0.0.1", port=8082)

        registry.register(service1)
        registry.register(service2)
        registry.register(service3)

        # 标记一个服务为不健康
        registry.mark_unhealthy("unhealthy")

        # 获取详细统计信息 (第305-307行)
        stats = registry.get_detailed_statistics()

        assert 'total_services' in stats
        assert 'healthy_services' in stats
        assert 'unhealthy_services' in stats
        assert 'service_names' in stats
        assert 'average_health_score' in stats

        assert stats['total_services'] == 3
        assert stats['healthy_services'] == 2
        assert stats['unhealthy_services'] == 1

    def test_connection_health_check_method(self):
        """测试连接健康检查方法 (第329行)"""
        registry = ServiceRegistry()

        service = ServiceInstance(id="conn_test", name="test", host="127.0.0.1", port=8080)
        registry.register(service)

        async def test_connection_health():
            # 模拟连接健康检查 (第329行)
            with patch('socket.socket') as mock_socket:
                mock_conn = Mock()
                mock_socket.return_value.__enter__.return_value = mock_conn

                result = await registry._connection_health_check(service)
                return result

        result = asyncio.run(test_connection_health())
        assert isinstance(result, bool)

    def test_health_checker_statistics(self):
        """测试健康检查器统计信息 (第352-355行)"""
        registry = ServiceRegistry()

        # 手动触发一些健康检查
        service = ServiceInstance(id="stats_test", name="test", host="127.0.0.1", port=8080)
        registry.register(service)

        # 获取健康检查器统计信息 (第352-355行)
        stats = registry.health_checker.get_statistics()

        assert 'total_checks' in stats
        assert 'successful_checks' in stats
        assert 'failed_checks' in stats
        assert 'check_interval' in stats  # 第352-355行
        assert 'timeout' in stats

    def test_health_check_with_connection_method(self):
        """测试使用连接方法的健康检查 (第359行)"""
        registry = ServiceRegistry()

        service = ServiceInstance(id="conn_health_test", name="test", host="127.0.0.1", port=8080)
        registry.register(service)

        async def test_health_with_connection():
            # 模拟连接健康检查 (第359行)
            with patch.object(registry, '_connection_health_check', return_value=True):
                result = await registry.health_checker._check_service_health(service)
                return result

        result = asyncio.run(test_health_with_connection())
        assert result is True

    def test_service_instance_comprehensive_info(self):
        """测试服务实例的全面信息 (第390行)"""
        service = ServiceInstance(
            id="comprehensive_test",
            name="test_service",
            host="127.0.0.1",
            port=8080,
            metadata={"version": "2.0", "environment": "production"},
            tags=["web", "api"],
            weight=2
        )

        # 获取全面信息 (第390行)
        info = service.get_comprehensive_info()

        assert info['id'] == "comprehensive_test"
        assert info['name'] == "test_service"
        assert info['host'] == "127.0.0.1"
        assert info['port'] == 8080
        assert info['metadata'] == {"version": "2.0", "environment": "production"}
        assert info['tags'] == ["web", "api"]
        assert info['weight'] == 2
        assert 'health_status' in info
        assert 'last_health_check' in info
        assert 'registration_time' in info

    async def test_async_health_check_flow(self):
        """测试异步健康检查流程"""
        registry = ServiceRegistry()

        service = ServiceInstance(id="async_test", name="test", host="127.0.0.1", port=8080)
        registry.register(service)

        # 启动健康检查
        registry.start_health_checking()

        # 等待一小段时间让健康检查运行
        await asyncio.sleep(0.1)

        # 停止健康检查
        registry.stop_health_checking()

        # 验证服务仍然存在
        assert registry.get_service("async_test") is not None

    def test_service_registration_with_validation(self):
        """测试服务注册与验证"""
        registry = ServiceRegistry()

        # 测试有效服务注册
        service = ServiceInstance(id="valid_test", name="test", host="127.0.0.1", port=8080)
        registry.register(service)

        assert registry.get_service("valid_test") is not None

        # 测试重复注册
        with pytest.raises(ValueError, match="Service with ID valid_test already exists"):
            registry.register(service)

        # 测试注销服务
        registry.unregister("valid_test")
        assert registry.get_service("valid_test") is None

        # 测试注销不存在的服务
        registry.unregister("nonexistent")  # 应该不抛出异常 (第420行)

    def test_service_selection_strategies(self):
        """测试服务选择策略"""
        registry = ServiceRegistry()

        # 添加多个服务
        services = []
        for i in range(3):
            service = ServiceInstance(
                id=f"service_{i}",
                name="test",
                host="127.0.0.1",
                port=8080 + i,
                weight=i + 1
            )
            services.append(service)
            registry.register(service)

        # 测试随机选择 (第560行)
        selected = registry.get_service_random("test")
        assert selected is not None
        assert selected.name == "test"

        # 测试轮询选择
        for i in range(3):
            selected = registry.get_service_round_robin("test")
            assert selected is not None
            assert selected.name == "test"

        # 测试加权选择
        selected = registry.get_service_weighted("test")
        assert selected is not None
        assert selected.name == "test"

        # 测试选择不存在服务的fallback (第571行)
        selected = registry.get_service_round_robin("nonexistent")
        assert selected is None

    def test_service_filtering_and_search(self):
        """测试服务过滤和搜索"""
        registry = ServiceRegistry()

        # 添加不同类型的服务
        web_service = ServiceInstance(
            id="web1",
            name="web",
            host="127.0.0.1",
            port=8080,
            tags=["web", "frontend"],
            metadata={"version": "1.0"}
        )
        api_service = ServiceInstance(
            id="api1",
            name="api",
            host="127.0.0.1",
            port=8081,
            tags=["api", "backend"],
            metadata={"version": "2.0"}
        )
        another_web = ServiceInstance(
            id="web2",
            name="web",
            host="127.0.0.1",
            port=8082,
            tags=["web", "frontend"],
            metadata={"version": "1.1"}
        )

        registry.register(web_service)
        registry.register(api_service)
        registry.register(another_web)

        # 测试按名称过滤
        web_services = registry.get_services_by_name("web")
        assert len(web_services) == 2
        assert all(service.name == "web" for service in web_services)

        # 测试按标签过滤
        frontend_services = registry.get_services_by_tag("frontend")
        assert len(frontend_services) == 2

        # 测试按元数据过滤
        v1_services = registry.get_services_by_metadata("version", "1.0")
        assert len(v1_services) == 1
        assert v1_services[0].id == "web1"

    def test_health_check_monitoring_lifecycle(self):
        """测试健康检查监控生命周期"""
        registry = ServiceRegistry()

        # 添加服务
        service = ServiceInstance(id="lifecycle_test", name="test", host="127.0.0.1", port=8080)
        registry.register(service)

        # 启动健康检查
        registry.start_health_checking()
        assert registry.is_running

        # 获取运行时统计
        runtime_stats = registry.get_runtime_statistics()
        assert 'is_running' in runtime_stats  # 测试第560行附近的功能
        assert runtime_stats['is_running'] is True

        # 停止健康检查
        registry.stop_health_checking()
        assert not registry.is_running

    def test_service_update_and_maintenance(self):
        """测试服务更新和维护"""
        registry = ServiceRegistry()

        # 注册服务
        service = ServiceInstance(
            id="update_test",
            name="test",
            host="127.0.0.1",
            port=8080,
            metadata={"version": "1.0"}
        )
        registry.register(service)

        # 更新服务元数据
        new_metadata = {"version": "1.1", "environment": "production"}
        registry.update_service_metadata("update_test", new_metadata)

        updated_service = registry.get_service("update_test")
        assert updated_service.metadata == new_metadata

        # 测试更新不存在服务
        registry.update_service_metadata("nonexistent", {"key": "value"})  # 应该不抛出异常 (第435行)

        # 测试服务权重更新
        registry.update_service_weight("update_test", 5)
        updated_service = registry.get_service("update_test")
        assert updated_service.weight == 5

    def test_batch_operations(self):
        """测试批量操作"""
        registry = ServiceRegistry()

        # 批量注册服务
        services = [
            ServiceInstance(id=f"batch_{i}", name="batch", host="127.0.0.1", port=8080 + i)
            for i in range(5)
        ]

        registry.register_services_batch(services)

        # 验证所有服务都被注册
        batch_services = registry.get_services_by_name("batch")
        assert len(batch_services) == 5

        # 批量注销
        service_ids = [f"batch_{i}" for i in range(5)]
        registry.unregister_services_batch(service_ids)

        # 验证所有服务都被注销
        batch_services_after = registry.get_services_by_name("batch")
        assert len(batch_services_after) == 0

    def test_error_handling_and_edge_cases(self):
        """测试错误处理和边缘情况"""
        registry = ServiceRegistry()

        # 测试空注册表的各种操作
        assert registry.get_service("nonexistent") is None
        assert registry.get_services_by_name("nonexistent") == []
        assert registry.get_services_by_tag("nonexistent") == []
        assert registry.get_service_random("nonexistent") is None
        assert registry.get_service_round_robin("nonexistent") is None
        assert registry.get_service_weighted("nonexistent") is None

        # 测试无效操作
        registry.unregister("nonexistent")  # 不应该抛出异常
        registry.mark_unhealthy("nonexistent")  # 不应该抛出异常
        registry.mark_healthy("nonexistent")  # 不应该抛出异常

        # 获取空注册表的统计信息
        stats = registry.get_statistics()
        assert stats['total_services'] == 0
        assert stats['healthy_services'] == 0

        detailed_stats = registry.get_detailed_statistics()
        assert detailed_stats['total_services'] == 0
        assert detailed_stats['service_names'] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])