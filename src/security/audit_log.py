"""
审计日志系统
提供全面的审计功能，包括事件记录、搜索、统计和导出
遵循TDD方法论开发，确保高测试覆盖率和可靠性
"""

import json
import uuid
import time
import threading
import csv
import io
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque


class AuditSeverity(Enum):
    """审计事件严重级别"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AuditCategory(Enum):
    """审计事件分类"""
    AUTHENTICATION = "authentication"
    USER_MANAGEMENT = "user_management"
    DATA_ACCESS = "data_access"
    SECURITY = "security"
    SYSTEM = "system"
    ADMINISTRATION = "administration"


class AuditEventType(Enum):
    """审计事件类型"""
    USER_ACTION = "user_action"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    user_id: str
    action: str
    resource: str
    severity: AuditSeverity = AuditSeverity.INFO
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: AuditCategory = field(init=False)
    event_type: AuditEventType = field(init=False)

    def __post_init__(self):
        """自动分类事件"""
        self.category = self._categorize_event()
        self.event_type = self._classify_event_type()

    def _categorize_event(self) -> AuditCategory:
        """根据动作和资源自动分类事件"""
        auth_actions = {"login", "logout", "authenticate", "verify_password"}
        user_actions = {"create_user", "update_user", "delete_user", "update_profile", "password_change"}
        data_actions = {"read", "write", "update", "delete", "export", "import", "access"}
        security_actions = {"privilege_escalation", "unauthorized_access", "security_breach", "audit"}
        admin_actions = {"admin_action", "system_config", "backup", "restore"}

        action_lower = self.action.lower()
        resource_lower = self.resource.lower()

        # 系统资源优先分类为系统事件
        if resource_lower == "system":
            return AuditCategory.SYSTEM
        elif action_lower in auth_actions or "auth" in resource_lower:
            return AuditCategory.AUTHENTICATION
        elif action_lower in user_actions or "user" in resource_lower:
            return AuditCategory.USER_MANAGEMENT
        elif action_lower in data_actions or "data" in resource_lower:
            return AuditCategory.DATA_ACCESS
        elif action_lower in security_actions or "security" in resource_lower:
            return AuditCategory.SECURITY
        elif action_lower in admin_actions or "admin" in resource_lower:
            return AuditCategory.ADMINISTRATION
        else:
            return AuditCategory.SYSTEM

    def _classify_event_type(self) -> AuditEventType:
        """分类事件类型"""
        security_indicators = {
            "privilege_escalation", "unauthorized_access", "security_breach",
            "failed_login", "suspicious_activity", "attack_detected",
            "password_change", "security_event"
        }
        system_indicators = {
            "startup", "shutdown", "backup", "restore", "maintenance",
            "system_config", "cron_job", "batch_process"
        }

        action_lower = self.action.lower()
        resource_lower = self.resource.lower()

        # 高严重级别且涉及密码/认证相关操作归类为安全事件
        if (self.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL] and
            ("password" in action_lower or "auth" in resource_lower or "security" in resource_lower)):
            return AuditEventType.SECURITY_EVENT

        if any(indicator in action_lower for indicator in security_indicators):
            return AuditEventType.SECURITY_EVENT
        elif any(indicator in action_lower for indicator in system_indicators):
            return AuditEventType.SYSTEM_EVENT
        else:
            return AuditEventType.USER_ACTION

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "severity": self.severity.value,
            "category": self.category.value,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "additional_data": self.additional_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """从字典创建事件"""
        # 处理时间戳，支持ISO格式和Z后缀格式
        timestamp_str = data["timestamp"]
        if timestamp_str.endswith('Z'):
            # 处理ISO 8601的Z格式
            timestamp_str = timestamp_str.replace('Z', '+00:00')

        timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # 处理枚举类型
        severity = AuditSeverity(data["severity"])

        event = cls(
            event_id=data["event_id"],
            user_id=data["user_id"],
            action=data["action"],
            resource=data["resource"],
            severity=severity,
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            additional_data=data.get("additional_data", {}),
            timestamp=timestamp
        )

        # 设置分类字段（由__post_init__处理）
        return event


@dataclass
class AuditLog:
    """审计日志记录"""
    log_id: str
    event: AuditEvent
    status: str  # success, failed, error
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "log_id": self.log_id,
            "event": self.event.to_dict(),
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "error_details": self.error_details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLog':
        """从字典创建日志记录"""
        # 处理时间戳，支持ISO格式和Z后缀格式
        timestamp_str = data["timestamp"]
        if timestamp_str.endswith('Z'):
            # 处理ISO 8601的Z格式
            timestamp_str = timestamp_str.replace('Z', '+00:00')

        timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        event = AuditEvent.from_dict(data["event"])

        return cls(
            log_id=data["log_id"],
            event=event,
            status=data["status"],
            message=data["message"],
            timestamp=timestamp,
            error_details=data.get("error_details")
        )


@dataclass
class AuditSearchFilter:
    """审计搜索过滤器"""
    user_ids: Optional[List[str]] = None
    actions: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    severities: Optional[List[AuditSeverity]] = None
    categories: Optional[List[AuditCategory]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None
    additional_data_filters: Optional[Dict[str, Any]] = None


@dataclass
class AuditStatistics:
    """审计统计信息"""
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    unique_users: int = 0
    critical_events: int = 0
    high_events: int = 0
    warning_events: int = 0
    info_events: int = 0
    events_by_category: Dict[str, int] = field(default_factory=dict)
    events_by_action: Dict[str, int] = field(default_factory=dict)
    top_active_users: List[Dict[str, Any]] = field(default_factory=list)


class AuditLogManager:
    """审计日志管理器"""

    def __init__(self, max_memory_events: int = 10000):
        """
        初始化审计日志管理器

        Args:
            max_memory_events: 内存中保存的最大事件数量
        """
        self.max_memory_events = max_memory_events
        self.memory_events: deque = deque(maxlen=max_memory_events)
        self.total_events_logged = 0
        self._lock = threading.RLock()
        self._statistics_cache: Optional[AuditStatistics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = 300  # 缓存5分钟

    def log_event(
        self,
        event: AuditEvent,
        status: str = "success",
        message: str = "",
        error_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录审计事件

        Args:
            event: 审计事件
            status: 事件状态 (success, failed, error)
            message: 事件消息
            error_details: 错误详情

        Returns:
            日志ID
        """
        log_id = f"log_{uuid.uuid4().hex[:8]}"

        log = AuditLog(
            log_id=log_id,
            event=event,
            status=status,
            message=message,
            error_details=error_details
        )

        with self._lock:
            self.memory_events.append(log)
            self.total_events_logged += 1
            self._invalidate_cache()

        return log_id

    def search_events(self, filter_criteria: AuditSearchFilter) -> List[AuditLog]:
        """
        搜索审计事件

        Args:
            filter_criteria: 搜索过滤条件

        Returns:
            匹配的日志记录列表
        """
        with self._lock:
            events = list(self.memory_events)

        # 应用过滤条件
        filtered_events = []
        for log in events:
            if self._matches_filter(log, filter_criteria):
                filtered_events.append(log)

        return filtered_events

    def _matches_filter(self, log: AuditLog, filter_criteria: AuditSearchFilter) -> bool:
        """检查日志是否匹配过滤条件"""
        event = log.event

        # 用户ID过滤
        if filter_criteria.user_ids and event.user_id not in filter_criteria.user_ids:
            return False

        # 动作过滤
        if filter_criteria.actions and event.action not in filter_criteria.actions:
            return False

        # 资源过滤
        if filter_criteria.resources and event.resource not in filter_criteria.resources:
            return False

        # 严重级别过滤
        if filter_criteria.severities and event.severity not in filter_criteria.severities:
            return False

        # 分类过滤
        if filter_criteria.categories and event.category not in filter_criteria.categories:
            return False

        # 时间范围过滤
        if filter_criteria.start_time and event.timestamp < filter_criteria.start_time:
            return False

        if filter_criteria.end_time and event.timestamp > filter_criteria.end_time:
            return False

        # 状态过滤
        if filter_criteria.status and log.status != filter_criteria.status:
            return False

        # 附加数据过滤
        if filter_criteria.additional_data_filters:
            for key, value in filter_criteria.additional_data_filters.items():
                if event.additional_data.get(key) != value:
                    return False

        return True

    def get_audit_statistics(self) -> AuditStatistics:
        """获取审计统计信息"""
        with self._lock:
            current_time = datetime.now(timezone.utc)

            # 检查缓存是否有效
            if (self._statistics_cache and
                self._cache_timestamp and
                (current_time - self._cache_timestamp).seconds < self._cache_ttl):
                return self._statistics_cache

            events = list(self.memory_events)

            stats = AuditStatistics()
            stats.total_events = len(events)

            # 统计基础数据
            users = set()
            events_by_category = defaultdict(int)
            events_by_action = defaultdict(int)
            user_activity = defaultdict(int)

            severity_counts = {
                AuditSeverity.CRITICAL: 0,
                AuditSeverity.HIGH: 0,
                AuditSeverity.WARNING: 0,
                AuditSeverity.INFO: 0
            }

            for log in events:
                # 状态统计
                if log.status == "success":
                    stats.successful_events += 1
                else:
                    stats.failed_events += 1

                # 用户统计
                users.add(log.event.user_id)
                user_activity[log.event.user_id] += 1

                # 严重级别统计
                severity_counts[log.event.severity] += 1

                # 分类统计
                events_by_category[log.event.category.value] += 1
                events_by_action[log.event.action] += 1

            # 设置统计结果
            stats.unique_users = len(users)
            stats.critical_events = severity_counts[AuditSeverity.CRITICAL]
            stats.high_events = severity_counts[AuditSeverity.HIGH]
            stats.warning_events = severity_counts[AuditSeverity.WARNING]
            stats.info_events = severity_counts[AuditSeverity.INFO]
            stats.events_by_category = dict(events_by_category)
            stats.events_by_action = dict(events_by_action)

            # 获取最活跃用户
            top_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            stats.top_active_users = [
                {"user_id": user_id, "event_count": count}
                for user_id, count in top_users
            ]

            # 更新缓存
            self._statistics_cache = stats
            self._cache_timestamp = current_time

            return stats

    def _invalidate_cache(self):
        """使缓存失效"""
        self._statistics_cache = None
        self._cache_timestamp = None

    def export_logs_to_json(self, filter_criteria: Optional[AuditSearchFilter] = None) -> str:
        """
        导出日志为JSON格式

        Args:
            filter_criteria: 可选的过滤条件

        Returns:
            JSON字符串
        """
        if filter_criteria:
            events = self.search_events(filter_criteria)
        else:
            with self._lock:
                events = list(self.memory_events)

        return json.dumps([log.to_dict() for log in events], indent=2)

    def import_logs_from_json(self, json_data: str) -> int:
        """
        从JSON导入日志

        Args:
            json_data: JSON字符串

        Returns:
            导入的日志数量
        """
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON data")

        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of audit logs")

        imported_count = 0
        with self._lock:
            for log_data in data:
                try:
                    log = AuditLog.from_dict(log_data)
                    self.memory_events.append(log)
                    self.total_events_logged += 1
                    imported_count += 1
                except (KeyError, ValueError, TypeError) as e:
                    # 跳过无效的日志记录
                    continue

            self._invalidate_cache()

        return imported_count

    def clear_old_events(self, cutoff_time: datetime) -> int:
        """
        清理旧事件

        Args:
            cutoff_time: 截止时间，早于此时间的事件将被删除

        Returns:
            删除的事件数量
        """
        with self._lock:
            original_count = len(self.memory_events)

            # 过滤掉旧事件
            filtered_events = deque(
                [log for log in self.memory_events if log.timestamp >= cutoff_time],
                maxlen=self.max_memory_events
            )

            self.memory_events = filtered_events
            deleted_count = original_count - len(self.memory_events)

            if deleted_count > 0:
                self._invalidate_cache()

        return deleted_count

    def get_user_activity_summary(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户活动摘要

        Args:
            user_id: 用户ID

        Returns:
            用户活动摘要
        """
        filter_criteria = AuditSearchFilter(user_ids=[user_id])
        user_events = self.search_events(filter_criteria)

        if not user_events:
            return {
                "user_id": user_id,
                "total_events": 0,
                "successful_events": 0,
                "failed_events": 0,
                "last_activity": None,
                "most_common_actions": []
            }

        successful_events = sum(1 for log in user_events if log.status == "success")
        failed_events = len(user_events) - successful_events
        last_activity = max(log.timestamp for log in user_events)

        # 统计最常见动作
        action_counts = defaultdict(int)
        for log in user_events:
            action_counts[log.event.action] += 1

        most_common_actions = [
            {"action": action, "count": count}
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        ][:5]

        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "successful_events": successful_events,
            "failed_events": failed_events,
            "last_activity": last_activity,
            "most_common_actions": most_common_actions
        }

    def get_security_events(self) -> List[AuditLog]:
        """获取安全相关事件"""
        security_categories = [
            AuditCategory.SECURITY,
            AuditCategory.AUTHENTICATION
        ]

        filter_criteria = AuditSearchFilter(categories=security_categories)
        return self.search_events(filter_criteria)


class AuditExporter:
    """审计数据导出器"""

    def export_to_csv(self, logs: List[AuditLog]) -> str:
        """
        导出日志为CSV格式

        Args:
            logs: 日志记录列表

        Returns:
            CSV字符串
        """
        output = io.StringIO()

        fieldnames = [
            'log_id', 'event_id', 'user_id', 'action', 'resource',
            'severity', 'category', 'status', 'message', 'timestamp',
            'ip_address', 'user_agent'
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            row = {
                'log_id': log.log_id,
                'event_id': log.event.event_id,
                'user_id': log.event.user_id,
                'action': log.event.action,
                'resource': log.event.resource,
                'severity': log.event.severity.value,
                'category': log.event.category.value,
                'status': log.status,
                'message': log.message,
                'timestamp': log.timestamp.isoformat(),
                'ip_address': log.event.ip_address or '',
                'user_agent': log.event.user_agent or ''
            }
            writer.writerow(row)

        return output.getvalue()


class AuditImporter:
    """审计数据导入器"""

    def import_from_json_file(self, file_path: str, manager: AuditLogManager) -> int:
        """
        从JSON文件导入日志

        Args:
            file_path: 文件路径
            manager: 审计日志管理器

        Returns:
            导入的日志数量
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()

            return manager.import_logs_from_json(json_data)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to import from file: {e}")


# 便利函数
def create_audit_event(
    user_id: str,
    action: str,
    resource: str,
    severity: AuditSeverity = AuditSeverity.INFO,
    **kwargs
) -> AuditEvent:
    """
    创建审计事件的便利函数

    Args:
        user_id: 用户ID
        action: 动作
        resource: 资源
        severity: 严重级别
        **kwargs: 其他参数

    Returns:
        审计事件
    """
    event_id = f"evt_{uuid.uuid4().hex[:8]}"

    return AuditEvent(
        event_id=event_id,
        user_id=user_id,
        action=action,
        resource=resource,
        severity=severity,
        **kwargs
    )


def log_security_event(
    manager: AuditLogManager,
    user_id: str,
    action: str,
    resource: str,
    severity: AuditSeverity = AuditSeverity.HIGH,
    **kwargs
) -> str:
    """
    记录安全事件的便利函数

    Args:
        manager: 审计日志管理器
        user_id: 用户ID
        action: 动作
        resource: 资源
        severity: 严重级别
        **kwargs: 其他参数

    Returns:
        日志ID
    """
    event = create_audit_event(
        user_id=user_id,
        action=action,
        resource=resource,
        severity=severity,
        **kwargs
    )

    return manager.log_event(
        event,
        status="success",
        message=f"Security event: {action}"
    )