"""
审计日志系统测试
遵循TDD方法论，确保100%测试通过率和95%以上代码覆盖率
"""

import pytest
import time
import json
import threading
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.security.audit_log import (
    AuditEvent, AuditLog, AuditLogManager, AuditSeverity,
    AuditCategory, AuditEventType, AuditSearchFilter,
    AuditStatistics, AuditExporter, AuditImporter
)


class TestAuditEvent:
    """审计事件测试类"""

    def test_create_basic_audit_event(self):
        """测试创建基础审计事件"""
        event = AuditEvent(
            event_id="evt_001",
            user_id="user_001",
            action="login",
            resource="auth",
            severity=AuditSeverity.INFO
        )

        assert event.event_id == "evt_001"
        assert event.user_id == "user_001"
        assert event.action == "login"
        assert event.resource == "auth"
        assert event.severity == AuditSeverity.INFO
        assert event.category == AuditCategory.AUTHENTICATION
        assert event.event_type == AuditEventType.USER_ACTION
        assert event.timestamp is not None
        assert event.ip_address is None
        assert event.user_agent is None
        assert event.additional_data == {}

    def test_create_full_audit_event(self):
        """测试创建完整的审计事件"""
        additional_data = {
            "login_method": "password",
            "failed_attempts": 3,
            "session_id": "sess_001"
        }

        event = AuditEvent(
            event_id="evt_002",
            user_id="user_002",
            action="password_change",
            resource="user_profile",
            severity=AuditSeverity.HIGH,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0...",
            additional_data=additional_data
        )

        assert event.event_id == "evt_002"
        assert event.additional_data == additional_data
        assert event.category == AuditCategory.USER_MANAGEMENT
        assert event.event_type == AuditEventType.SECURITY_EVENT

    def test_event_categorization(self):
        """测试事件自动分类"""
        # 认证相关事件
        auth_event = AuditEvent(
            event_id="evt_003",
            user_id="user_003",
            action="login",
            resource="auth"
        )
        assert auth_event.category == AuditCategory.AUTHENTICATION

        # 用户管理事件
        user_event = AuditEvent(
            event_id="evt_004",
            user_id="user_004",
            action="create_user",
            resource="users"
        )
        assert user_event.category == AuditCategory.USER_MANAGEMENT

        # 数据访问事件
        data_event = AuditEvent(
            event_id="evt_005",
            user_id="user_005",
            action="read",
            resource="sensitive_data"
        )
        assert data_event.category == AuditCategory.DATA_ACCESS

        # 系统事件
        system_event = AuditEvent(
            event_id="evt_006",
            user_id="system",
            action="backup",
            resource="system"
        )
        assert system_event.category == AuditCategory.SYSTEM

    def test_event_type_classification(self):
        """测试事件类型分类"""
        # 用户操作
        user_action = AuditEvent(
            event_id="evt_007",
            user_id="user_007",
            action="update_profile",
            resource="user"
        )
        assert user_action.event_type == AuditEventType.USER_ACTION

        # 安全事件
        security_event = AuditEvent(
            event_id="evt_008",
            user_id="user_008",
            action="privilege_escalation",
            resource="system"
        )
        assert security_event.event_type == AuditEventType.SECURITY_EVENT

        # 系统事件
        system_event = AuditEvent(
            event_id="evt_009",
            user_id="system",
            action="startup",
            resource="system"
        )
        assert system_event.event_type == AuditEventType.SYSTEM_EVENT

    def test_event_to_dict(self):
        """测试事件转换为字典"""
        event = AuditEvent(
            event_id="evt_010",
            user_id="user_010",
            action="login",
            resource="auth",
            severity=AuditSeverity.WARNING,
            ip_address="10.0.0.1",
            user_agent="Test Agent",
            additional_data={"key": "value"}
        )

        event_dict = event.to_dict()

        assert event_dict["event_id"] == "evt_010"
        assert event_dict["user_id"] == "user_010"
        assert event_dict["action"] == "login"
        assert event_dict["resource"] == "auth"
        assert event_dict["severity"] == "warning"
        assert event_dict["category"] == "authentication"
        assert event_dict["event_type"] == "user_action"
        assert event_dict["ip_address"] == "10.0.0.1"
        assert event_dict["user_agent"] == "Test Agent"
        assert event_dict["additional_data"] == {"key": "value"}
        assert "timestamp" in event_dict

    def test_event_from_dict(self):
        """测试从字典创建事件"""
        event_data = {
            "event_id": "evt_011",
            "user_id": "user_011",
            "action": "logout",
            "resource": "auth",
            "severity": "info",
            "category": "authentication",
            "event_type": "user_action",
            "timestamp": "2024-01-01T12:00:00Z",
            "ip_address": "192.168.1.1",
            "user_agent": "Browser",
            "additional_data": {"session_id": "sess_123"}
        }

        event = AuditEvent.from_dict(event_data)

        assert event.event_id == "evt_011"
        assert event.user_id == "user_011"
        assert event.action == "logout"
        assert event.resource == "auth"
        assert event.severity == AuditSeverity.INFO
        assert event.ip_address == "192.168.1.1"
        assert event.additional_data == {"session_id": "sess_123"}

    def test_event_serialization_roundtrip(self):
        """测试事件序列化往返"""
        original_event = AuditEvent(
            event_id="evt_012",
            user_id="user_012",
            action="data_export",
            resource="analytics",
            severity=AuditSeverity.HIGH,
            additional_data={"export_format": "csv", "record_count": 1000}
        )

        # 转换为字典
        event_dict = original_event.to_dict()

        # 从字典重建
        reconstructed_event = AuditEvent.from_dict(event_dict)

        # 验证一致性
        assert reconstructed_event.event_id == original_event.event_id
        assert reconstructed_event.user_id == original_event.user_id
        assert reconstructed_event.action == original_event.action
        assert reconstructed_event.resource == original_event.resource
        assert reconstructed_event.severity == original_event.severity
        assert reconstructed_event.additional_data == original_event.additional_data


class TestAuditLog:
    """审计日志测试类"""

    def test_create_audit_log(self):
        """测试创建审计日志"""
        log = AuditLog(
            log_id="log_001",
            event=AuditEvent(
                event_id="evt_001",
                user_id="user_001",
                action="login",
                resource="auth"
            ),
            status="success",
            message="Login successful"
        )

        assert log.log_id == "log_001"
        assert log.event.event_id == "evt_001"
        assert log.status == "success"
        assert log.message == "Login successful"
        assert log.timestamp is not None
        assert log.error_details is None

    def test_create_audit_log_with_error(self):
        """测试创建带错误的审计日志"""
        log = AuditLog(
            log_id="log_002",
            event=AuditEvent(
                event_id="evt_002",
                user_id="user_002",
                action="login",
                resource="auth"
            ),
            status="failed",
            message="Login failed",
            error_details={
                "error_code": "INVALID_PASSWORD",
                "error_message": "Invalid password provided",
                "failed_attempts": 3
            }
        )

        assert log.status == "failed"
        assert log.error_details["error_code"] == "INVALID_PASSWORD"
        assert log.error_details["failed_attempts"] == 3

    def test_log_to_dict(self):
        """测试日志转换为字典"""
        event = AuditEvent(
            event_id="evt_003",
            user_id="user_003",
            action="password_change",
            resource="user"
        )

        log = AuditLog(
            log_id="log_003",
            event=event,
            status="success",
            message="Password changed successfully"
        )

        log_dict = log.to_dict()

        assert log_dict["log_id"] == "log_003"
        assert log_dict["event"]["event_id"] == "evt_003"
        assert log_dict["status"] == "success"
        assert log_dict["message"] == "Password changed successfully"
        assert "timestamp" in log_dict
        assert log_dict["error_details"] is None

    def test_log_from_dict(self):
        """测试从字典创建日志"""
        log_data = {
            "log_id": "log_004",
            "event": {
                "event_id": "evt_004",
                "user_id": "user_004",
                "action": "data_access",
                "resource": "files",
                "severity": "info",
                "category": "data_access",
                "event_type": "user_action",
                "timestamp": "2024-01-01T12:00:00Z",
                "ip_address": "10.0.0.1",
                "user_agent": "Test Agent",
                "additional_data": {}
            },
            "status": "success",
            "message": "File accessed successfully",
            "timestamp": "2024-01-01T12:00:01Z",
            "error_details": None
        }

        log = AuditLog.from_dict(log_data)

        assert log.log_id == "log_004"
        assert log.event.event_id == "evt_004"
        assert log.status == "success"
        assert log.message == "File accessed successfully"
        assert log.error_details is None


class TestAuditLogManager:
    """审计日志管理器测试类"""

    def test_create_manager(self):
        """测试创建管理器"""
        manager = AuditLogManager(max_memory_events=1000)

        assert manager.max_memory_events == 1000
        assert len(manager.memory_events) == 0
        assert manager.total_events_logged == 0

    def test_log_event_success(self):
        """测试记录成功事件"""
        manager = AuditLogManager()

        event = AuditEvent(
            event_id="evt_001",
            user_id="user_001",
            action="login",
            resource="auth"
        )

        log_id = manager.log_event(event, status="success", message="Login successful")

        assert log_id is not None
        assert manager.total_events_logged == 1
        assert len(manager.memory_events) == 1

        stored_log = manager.memory_events[0]
        assert stored_log.event.event_id == "evt_001"
        assert stored_log.status == "success"
        assert stored_log.message == "Login successful"

    def test_log_event_with_error(self):
        """测试记录错误事件"""
        manager = AuditLogManager()

        event = AuditEvent(
            event_id="evt_002",
            user_id="user_002",
            action="login",
            resource="auth"
        )

        error_details = {"error_code": "INVALID_CREDENTIALS"}
        log_id = manager.log_event(
            event,
            status="failed",
            message="Login failed",
            error_details=error_details
        )

        assert log_id is not None
        assert manager.total_events_logged == 1

        stored_log = manager.memory_events[0]
        assert stored_log.status == "failed"
        assert stored_log.error_details == error_details

    def test_memory_events_limit(self):
        """测试内存事件数量限制"""
        manager = AuditLogManager(max_memory_events=3)

        # 添加超过限制的事件
        for i in range(5):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action="test_action",
                resource="test"
            )
            manager.log_event(event)

        # 验证只保留最新的事件
        assert len(manager.memory_events) == 3
        assert manager.memory_events[0].event.event_id == "evt_002"
        assert manager.memory_events[2].event.event_id == "evt_004"
        assert manager.total_events_logged == 5

    def test_search_events_by_user(self):
        """测试按用户搜索事件"""
        manager = AuditLogManager()

        # 创建不同用户的事件
        events_data = [
            ("user_001", "login"),
            ("user_002", "login"),
            ("user_001", "data_access"),
            ("user_003", "login"),
            ("user_001", "logout")
        ]

        for user_id, action in events_data:
            event = AuditEvent(
                event_id=f"evt_{user_id}_{action}",
                user_id=user_id,
                action=action,
                resource="test"
            )
            manager.log_event(event)

        # 搜索user_001的事件
        filter_criteria = AuditSearchFilter(user_ids=["user_001"])
        results = manager.search_events(filter_criteria)

        assert len(results) == 3
        for log in results:
            assert log.event.user_id == "user_001"

    def test_search_events_by_action(self):
        """测试按动作搜索事件"""
        manager = AuditLogManager()

        actions = ["login", "logout", "login", "data_access", "login"]
        for i, action in enumerate(actions):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action=action,
                resource="test"
            )
            manager.log_event(event)

        # 搜索登录事件
        filter_criteria = AuditSearchFilter(actions=["login"])
        results = manager.search_events(filter_criteria)

        assert len(results) == 3
        for log in results:
            assert log.event.action == "login"

    def test_search_events_by_severity(self):
        """测试按严重级别搜索事件"""
        manager = AuditLogManager()

        severities = [
            AuditSeverity.INFO,
            AuditSeverity.WARNING,
            AuditSeverity.HIGH,
            AuditSeverity.CRITICAL,
            AuditSeverity.INFO
        ]

        for i, severity in enumerate(severities):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action="test",
                resource="test",
                severity=severity
            )
            manager.log_event(event)

        # 搜索高级别事件
        filter_criteria = AuditSearchFilter(
            severities=[AuditSeverity.HIGH, AuditSeverity.CRITICAL]
        )
        results = manager.search_events(filter_criteria)

        assert len(results) == 2

    def test_search_events_by_time_range(self):
        """测试按时间范围搜索事件"""
        manager = AuditLogManager()

        base_time = datetime.now(timezone.utc)

        # 创建不同时间的事件
        times = [
            base_time,
            base_time.replace(second=10),
            base_time.replace(second=20),
            base_time.replace(second=30),
            base_time.replace(second=40)
        ]

        for i, event_time in enumerate(times):
            # 直接创建带时间戳的事件
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action="test",
                resource="test",
                timestamp=event_time
            )
            manager.log_event(event)

        # 搜索中间时间范围的事件
        start_time = base_time.replace(second=15)
        end_time = base_time.replace(second=35)

        filter_criteria = AuditSearchFilter(
            start_time=start_time,
            end_time=end_time
        )
        results = manager.search_events(filter_criteria)

        assert len(results) == 2  # evt_001, evt_002

    def test_search_events_combined_filters(self):
        """测试组合搜索条件"""
        manager = AuditLogManager()

        # 创建测试数据
        test_data = [
            ("user_001", "login", AuditSeverity.INFO),
            ("user_001", "data_access", AuditSeverity.HIGH),
            ("user_002", "login", AuditSeverity.INFO),
            ("user_001", "login", AuditSeverity.WARNING),
            ("user_001", "admin_action", AuditSeverity.HIGH)
        ]

        for i, (user_id, action, severity) in enumerate(test_data):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id=user_id,
                action=action,
                resource="test",
                severity=severity
            )
            manager.log_event(event)

        # 组合搜索：user_001的高级事件
        filter_criteria = AuditSearchFilter(
            user_ids=["user_001"],
            severities=[AuditSeverity.HIGH]
        )
        results = manager.search_events(filter_criteria)

        assert len(results) == 2
        for log in results:
            assert log.event.user_id == "user_001"
            assert log.event.severity == AuditSeverity.HIGH

    def test_get_audit_statistics(self):
        """测试获取审计统计信息"""
        manager = AuditLogManager()

        # 创建不同类型的事件
        events = [
            ("user_001", "login", AuditSeverity.INFO, "success"),
            ("user_001", "login", AuditSeverity.WARNING, "failed"),
            ("user_002", "data_access", AuditSeverity.HIGH, "success"),
            ("user_001", "admin_action", AuditSeverity.CRITICAL, "success"),
            ("user_003", "login", AuditSeverity.INFO, "failed")
        ]

        for user_id, action, severity, status in events:
            event = AuditEvent(
                event_id=f"evt_{user_id}_{action}",
                user_id=user_id,
                action=action,
                resource="test",
                severity=severity
            )
            manager.log_event(event, status=status)

        stats = manager.get_audit_statistics()

        assert stats.total_events == 5
        assert stats.successful_events == 3
        assert stats.failed_events == 2
        assert stats.unique_users == 3
        assert stats.critical_events == 1
        assert stats.high_events == 1
        assert stats.warning_events == 1
        assert stats.info_events == 2

    def test_export_logs_to_json(self):
        """测试导出日志为JSON"""
        manager = AuditLogManager()

        # 创建测试事件
        for i in range(3):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action="test_action",
                resource="test"
            )
            manager.log_event(event)

        # 导出为JSON
        json_data = manager.export_logs_to_json()

        assert isinstance(json_data, str)

        # 解析并验证
        parsed_data = json.loads(json_data)
        assert len(parsed_data) == 3
        assert parsed_data[0]["event"]["event_id"] == "evt_000"

    def test_import_logs_from_json(self):
        """测试从JSON导入日志"""
        manager = AuditLogManager()

        # 准备测试数据
        test_logs = [
            {
                "log_id": "log_001",
                "event": {
                    "event_id": "evt_001",
                    "user_id": "user_001",
                    "action": "login",
                    "resource": "auth",
                    "severity": "info",
                    "category": "authentication",
                    "event_type": "user_action",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "ip_address": None,
                    "user_agent": None,
                    "additional_data": {}
                },
                "status": "success",
                "message": "Login successful",
                "timestamp": "2024-01-01T12:00:01Z",
                "error_details": None
            }
        ]

        json_data = json.dumps(test_logs)
        imported_count = manager.import_logs_from_json(json_data)

        assert imported_count == 1
        assert len(manager.memory_events) == 1
        assert manager.memory_events[0].event.event_id == "evt_001"

    def test_concurrent_logging(self):
        """测试并发日志记录"""
        manager = AuditLogManager()
        results = []

        def log_events(thread_id: int, count: int):
            for i in range(count):
                event = AuditEvent(
                    event_id=f"evt_{thread_id}_{i}",
                    user_id=f"user_{thread_id}",
                    action="test_action",
                    resource="test"
                )
                log_id = manager.log_event(event)
                results.append(log_id)

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_events, args=(i, 5))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 15
        assert manager.total_events_logged == 15
        assert len(manager.memory_events) == 15  # 默认内存限制足够大

    def test_clear_old_events(self):
        """测试清理旧事件"""
        manager = AuditLogManager(max_memory_events=100)

        # 创建不同时间的事件
        base_time = datetime.now(timezone.utc)

        for i in range(10):
            event_time = base_time.replace(hour=i)
            with patch('src.security.audit_log.datetime') as mock_datetime:
                mock_datetime.now.return_value = event_time

                event = AuditEvent(
                    event_id=f"evt_{i:03d}",
                    user_id="user_001",
                    action="test",
                    resource="test"
                )
                manager.log_event(event)

        # 清理5小时前的事件
        cutoff_time = base_time.replace(hour=5)
        cleared_count = manager.clear_old_events(cutoff_time)

        assert cleared_count == 5  # 清理了0-4点的事件
        assert len(manager.memory_events) == 5  # 剩余5-9点的事件

    def test_get_user_activity_summary(self):
        """测试获取用户活动摘要"""
        manager = AuditLogManager()

        # 创建用户活动数据
        activities = [
            ("user_001", "login", "success"),
            ("user_001", "data_access", "success"),
            ("user_002", "login", "failed"),
            ("user_001", "admin_action", "success"),
            ("user_001", "logout", "success"),
            ("user_002", "login", "success")
        ]

        for user_id, action, status in activities:
            event = AuditEvent(
                event_id=f"evt_{user_id}_{action}",
                user_id=user_id,
                action=action,
                resource="test"
            )
            manager.log_event(event, status=status)

        summary = manager.get_user_activity_summary("user_001")

        assert summary["user_id"] == "user_001"
        assert summary["total_events"] == 4
        assert summary["successful_events"] == 4
        assert summary["failed_events"] == 0
        assert summary["last_activity"] is not None

    def test_get_security_events(self):
        """测试获取安全相关事件"""
        manager = AuditLogManager()

        # 创建不同类型的事件
        events = [
            ("user_001", "login", AuditSeverity.INFO, AuditCategory.AUTHENTICATION),
            ("user_001", "privilege_escalation", AuditSeverity.HIGH, AuditCategory.SECURITY),
            ("user_002", "data_access", AuditSeverity.WARNING, AuditCategory.DATA_ACCESS),
            ("user_003", "unauthorized_access", AuditSeverity.CRITICAL, AuditCategory.SECURITY),
            ("user_001", "logout", AuditSeverity.INFO, AuditCategory.AUTHENTICATION)
        ]

        for user_id, action, severity, category in events:
            event = AuditEvent(
                event_id=f"evt_{user_id}_{action}",
                user_id=user_id,
                action=action,
                resource="test",
                severity=severity
            )
            # 设置category
            event.category = category
            manager.log_event(event)

        security_events = manager.get_security_events()

        # 过滤安全相关事件
        security_related = [
            log for log in security_events
            if log.event.category in [AuditCategory.SECURITY, AuditCategory.AUTHENTICATION]
        ]

        assert len(security_related) >= 3  # 至少包含security和authentication事件


class TestAuditExporter:
    """审计导出器测试类"""

    def test_export_to_csv(self):
        """测试导出为CSV格式"""
        manager = AuditLogManager()

        # 创建测试数据
        for i in range(3):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action="test_action",
                resource="test",
                severity=AuditSeverity.INFO
            )
            manager.log_event(event)

        exporter = AuditExporter()
        csv_data = exporter.export_to_csv(manager.memory_events)

        assert isinstance(csv_data, str)
        lines = csv_data.strip().split('\n')
        assert len(lines) == 4  # 标题行 + 3行数据
        assert "event_id,user_id,action,resource" in lines[0]  # CSV标题

    def test_export_filtered_data(self):
        """测试导出过滤后的数据"""
        manager = AuditLogManager()

        # 创建不同严重级别的事件
        severities = [AuditSeverity.INFO, AuditSeverity.WARNING, AuditSeverity.HIGH]
        for i, severity in enumerate(severities):
            event = AuditEvent(
                event_id=f"evt_{i:03d}",
                user_id="user_001",
                action="test",
                resource="test",
                severity=severity
            )
            manager.log_event(event)

        # 只导出高级别事件
        filter_criteria = AuditSearchFilter(
            severities=[AuditSeverity.HIGH, AuditSeverity.WARNING]
        )
        filtered_logs = manager.search_events(filter_criteria)

        exporter = AuditExporter()
        csv_data = exporter.export_to_csv(filtered_logs)

        lines = csv_data.strip().split('\n')
        assert len(lines) == 3  # 标题行 + 2行过滤后的数据


class TestAuditImporter:
    """审计导入器测试类"""

    def test_import_from_json_file(self):
        """测试从JSON文件导入"""
        manager = AuditLogManager()

        # 准备测试JSON文件
        test_data = [
            {
                "log_id": "log_001",
                "event": {
                    "event_id": "evt_001",
                    "user_id": "user_001",
                    "action": "login",
                    "resource": "auth",
                    "severity": "info",
                    "category": "authentication",
                    "event_type": "user_action",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "ip_address": None,
                    "user_agent": None,
                    "additional_data": {}
                },
                "status": "success",
                "message": "Login successful",
                "timestamp": "2024-01-01T12:00:01Z",
                "error_details": None
            }
        ]

        from unittest.mock import mock_open
        json_str = json.dumps(test_data)

        with patch('builtins.open', mock_open(read_data=json_str)):
            importer = AuditImporter()
            imported_count = importer.import_from_json_file("test.json", manager)
            assert imported_count == 1
            assert len(manager.memory_events) == 1


def mock_open_json(data):
    """模拟打开JSON文件"""
    from unittest.mock import mock_open

    json_str = json.dumps(data)
    return mock_open(read_data=json_str)


class TestAuditIntegration:
    """审计系统集成测试"""

    def test_rbac_audit_integration(self):
        """测试RBAC系统集成"""
        from src.security.rbac import RBACManager, Permission

        manager = AuditLogManager()
        rbac = RBACManager()

        # 创建权限和角色
        rbac.create_permission("read:sensitive", "读取敏感数据")
        rbac.create_role("analyst", "数据分析师")
        rbac.assign_permission_to_role("analyst", "read:sensitive")

        # 创建用户并分配角色
        rbac.create_user("user_001", {"name": "Test User"})
        rbac.assign_role_to_user("user_001", "analyst")

        # 审计权限检查
        event = AuditEvent(
            event_id="evt_rbac_001",
            user_id="user_001",
            action="permission_check",
            resource="rbac",
            additional_data={
                "permission": "read:sensitive",
                "result": "granted"
            }
        )

        log_id = manager.log_event(
            event,
            status="success",
            message="Permission check completed"
        )

        assert log_id is not None
        assert len(manager.memory_events) == 1

        # 验证审计记录
        log = manager.memory_events[0]
        assert log.event.additional_data["permission"] == "read:sensitive"
        assert log.event.additional_data["result"] == "granted"

    def test_jwt_audit_integration(self):
        """测试JWT系统集成"""
        from src.security.jwt_auth import JWTAuthenticator

        manager = AuditLogManager()
        jwt_auth = JWTAuthenticator("test_secret")

        # 审计令牌生成
        event = AuditEvent(
            event_id="evt_jwt_001",
            user_id="user_001",
            action="token_generation",
            resource="jwt",
            additional_data={
                "token_type": "access",
                "expires_in": 3600
            }
        )

        log_id = manager.log_event(
            event,
            status="success",
            message="JWT token generated successfully"
        )

        assert log_id is not None

        # 验证审计记录
        log = manager.memory_events[0]
        assert log.event.action == "token_generation"
        assert log.event.additional_data["token_type"] == "access"

    def test_encryption_audit_integration(self):
        """测试加密系统集成"""
        from src.security.encryption import DataEncryption

        manager = AuditLogManager()
        encryption = DataEncryption()

        # 审计数据加密操作
        event = AuditEvent(
            event_id="evt_enc_001",
            user_id="user_001",
            action="data_encryption",
            resource="sensitive_data",
            severity=AuditSeverity.HIGH,
            additional_data={
                "algorithm": "AES",
                "data_type": "personal_info",
                "encryption_success": True
            }
        )

        log_id = manager.log_event(
            event,
            status="success",
            message="Data encrypted successfully"
        )

        assert log_id is not None

        # 验证审计记录
        log = manager.memory_events[0]
        assert log.event.severity == AuditSeverity.HIGH
        assert log.event.additional_data["algorithm"] == "AES"

    def test_security_workflow_audit(self):
        """测试完整安全工作流审计"""
        manager = AuditLogManager()

        workflow_id = "workflow_001"
        user_id = "user_001"

        # 模拟完整的安全工作流
        workflow_steps = [
            ("login", "auth", AuditSeverity.INFO, "User login successful"),
            ("permission_check", "rbac", AuditSeverity.INFO, "Permission verified"),
            ("data_access", "database", AuditSeverity.HIGH, "Sensitive data accessed"),
            ("data_encryption", "encryption", AuditSeverity.HIGH, "Data encrypted for export"),
            ("logout", "auth", AuditSeverity.INFO, "User logout successful")
        ]

        for i, (action, resource, severity, message) in enumerate(workflow_steps):
            event = AuditEvent(
                event_id=f"evt_{workflow_id}_{i:03d}",
                user_id=user_id,
                action=action,
                resource=resource,
                severity=severity,
                additional_data={
                    "workflow_id": workflow_id,
                    "step_number": i + 1,
                    "total_steps": len(workflow_steps)
                }
            )

            log_id = manager.log_event(
                event,
                status="success",
                message=message
            )

            assert log_id is not None

        # 验证完整工作流被记录
        assert len(manager.memory_events) == len(workflow_steps)

        # 搜索特定工作流的所有事件
        filter_criteria = AuditSearchFilter(
            additional_data_filters={
                "workflow_id": workflow_id
            }
        )

        workflow_events = manager.search_events(filter_criteria)
        assert len(workflow_events) == len(workflow_steps)

        # 验证事件顺序
        for i, log in enumerate(workflow_events):
            step_number = log.event.additional_data["step_number"]
            assert step_number == i + 1

    @patch('src.security.audit_log.datetime')
    def test_performance_audit_logging(self, mock_datetime):
        """测试性能审计日志记录"""
        manager = AuditLogManager()

        # 固定时间以确保一致性
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_time

        # 记录开始时间
        start_time = time.time()

        # 批量记录事件
        event_count = 1000
        for i in range(event_count):
            event = AuditEvent(
                event_id=f"evt_perf_{i:06d}",
                user_id=f"user_{i % 10}",  # 10个不同用户
                action="bulk_operation",
                resource="performance_test"
            )
            manager.log_event(event)

        end_time = time.time()
        duration = end_time - start_time

        # 性能验证
        assert duration < 5.0  # 应该在5秒内完成
        assert manager.total_events_logged == event_count
        assert len(manager.memory_events) == event_count

        # 验证性能统计
        events_per_second = event_count / duration
        assert events_per_second > 100  # 至少每秒100个事件


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])