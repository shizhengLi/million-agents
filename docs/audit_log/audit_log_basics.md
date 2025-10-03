# 审计日志基础知识

## 什么是审计日志

审计日志（Audit Log）是系统安全的核心组件，用于记录和分析系统中发生的所有重要事件。它提供了完整的操作追踪、合规监控和安全分析能力，是现代企业级系统不可或缺的基础设施。

## 审计日志的重要性

### 1. **安全合规**
- **法规要求**: GDPR、SOX、HIPAA等法规要求详细的操作记录
- **合规审计**: 满足内部和外部审计需求
- **责任追溯**: 明确操作责任和时间线

### 2. **安全监控**
- **威胁检测**: 识别异常行为和安全威胁
- **攻击分析**: 分析攻击路径和影响范围
- **事件响应**: 为安全事件提供调查依据

### 3. **业务分析**
- **用户行为**: 分析用户操作模式
- **系统使用**: 了解系统资源使用情况
- **业务洞察**: 提供业务决策数据支持

## 审计日志的核心概念

### 1. **事件（Event）**
审计的基本单位，记录单个操作或状态变化。

```python
# 事件示例
{
    "event_id": "evt_001",
    "timestamp": "2024-01-01T12:00:00Z",
    "user_id": "user_001",
    "action": "login",
    "resource": "auth",
    "severity": "info",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "additional_data": {
        "login_method": "password",
        "session_id": "sess_001"
    }
}
```

### 2. **日志记录（Log Entry）**
包含事件元数据的完整记录。

```python
# 日志记录示例
{
    "log_id": "log_001",
    "event": { /* 事件数据 */ },
    "status": "success",
    "message": "Login successful",
    "timestamp": "2024-01-01T12:00:01Z",
    "error_details": null
}
```

### 3. **分类（Category）**
按业务功能对事件进行分类。

#### 主要分类
- **认证（Authentication）**: 登录、登出、密码验证
- **用户管理（User Management）**: 用户创建、更新、删除
- **数据访问（Data Access）**: 数据读取、修改、删除
- **安全（Security）**: 权限变更、安全事件
- **系统（System）**: 系统配置、维护操作
- **管理（Administration）**: 管理员操作

```python
# 分类示例
event_categories = {
    "login": "authentication",
    "create_user": "user_management",
    "read_data": "data_access",
    "privilege_change": "security",
    "system_backup": "system",
    "admin_config": "administration"
}
```

### 4. **严重级别（Severity）**
表示事件的重要性和紧急程度。

#### 严重级别定义
- **INFO**: 一般信息，正常操作
- **WARNING**: 警告信息，需要关注
- **HIGH**: 高风险事件，需要立即处理
- **CRITICAL**: 严重事件，紧急处理

```python
# 严重级别示例
severity_examples = {
    "user_login": "info",
    "failed_login": "warning",
    "privilege_escalation": "high",
    "data_breach": "critical"
}
```

### 5. **事件类型（Event Type）**
根据事件性质进行分类。

#### 事件类型
- **用户操作（User Action）**: 正常的用户业务操作
- **安全事件（Security Event）**: 涉及安全的操作
- **系统事件（System Event）**: 系统级别的操作

## 审计日志的设计原则

### 1. **完整性原则**
```python
# ✅ 记录完整的上下文信息
audit_event = {
    "user_id": "user_001",
    "action": "data_access",
    "resource": "customer_records",
    "timestamp": "2024-01-01T12:00:00Z",
    "ip_address": "10.0.0.1",
    "user_agent": "API Client v1.0",
    "additional_data": {
        "query_params": {"limit": 100},
        "response_count": 85,
        "execution_time_ms": 150
    }
}
```

### 2. **不可篡改原则**
```python
# ✅ 使用不可变记录
@dataclass(frozen=True)
class AuditLog:
    log_id: str
    event: AuditEvent
    timestamp: datetime
    # 一旦创建就不能修改

# ✅ 添加数字签名（可选）
def sign_audit_log(log: AuditLog, private_key) -> str:
    log_data = json.dumps(log.to_dict(), sort_keys=True)
    signature = rsa.sign(log_data.encode(), private_key, 'SHA-256')
    return base64.b64encode(signature).decode()
```

### 3. **实时性原则**
```python
# ✅ 异步记录，不影响主业务流程
import asyncio

async def log_event_async(event: AuditEvent):
    """异步记录审计事件"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, audit_manager.log_event, event)
    # 不阻塞主业务逻辑

# 使用示例
async def handle_user_login(user_id: str):
    # 处理登录逻辑
    await authenticate_user(user_id)

    # 异步记录审计日志
    event = create_audit_event(user_id, "login", "auth")
    asyncio.create_task(log_event_async(event))
```

### 4. **可查询性原则**
```python
# ✅ 支持多维度查询
class AuditSearchFilter:
    user_ids: List[str] = None          # 按用户查询
    actions: List[str] = None           # 按动作查询
    resources: List[str] = None         # 按资源查询
    severities: List[AuditSeverity] = None  # 按严重级别查询
    start_time: datetime = None         # 时间范围查询
    end_time: datetime = None
    additional_data_filters: Dict = None  # 自定义过滤条件

# 查询示例
filter = AuditSearchFilter(
    user_ids=["user_001", "user_002"],
    severities=[AuditSeverity.HIGH, AuditSeverity.CRITICAL],
    start_time=datetime.now() - timedelta(hours=24)
)
```

## 高性能审计日志设计

### 1. **内存管理策略**
```python
from collections import deque
import threading

class HighPerformanceAuditLog:
    def __init__(self, max_events: int = 100000):
        # 使用deque实现循环缓冲区，自动删除旧事件
        self.events = deque(maxlen=max_events)
        self.lock = threading.RLock()  # 线程安全

    def add_event(self, event: AuditEvent):
        """添加事件，自动管理内存"""
        with self.lock:
            self.events.append(event)
            # 如果超出容量，最旧的事件会被自动删除

    def get_recent_events(self, count: int = 1000):
        """获取最近的事件"""
        with self.lock:
            return list(self.events)[-count:]
```

### 2. **批量处理策略**
```python
class BatchAuditLogger:
    def __init__(self, batch_size: int = 100, flush_interval: int = 5):
        self.batch = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()

    def log_event(self, event: AuditEvent):
        """批量记录事件"""
        self.batch.append(event)

        # 检查是否需要刷新
        if (len(self.batch) >= self.batch_size or
            time.time() - self.last_flush > self.flush_interval):
            self.flush_batch()

    def flush_batch(self):
        """刷新批次到存储"""
        if self.batch:
            # 批量写入存储
            storage.write_events(self.batch)
            self.batch.clear()
            self.last_flush = time.time()
```

### 3. **缓存策略**
```python
from functools import lru_cache
import time

class CachedAuditStatistics:
    def __init__(self, cache_ttl: int = 300):  # 5分钟缓存
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_timestamp = {}

    def get_statistics(self, manager) -> AuditStatistics:
        """获取带缓存的统计信息"""
        cache_key = "audit_stats"
        current_time = time.time()

        # 检查缓存是否有效
        if (cache_key in self._cache and
            cache_key in self._cache_timestamp and
            current_time - self._cache_timestamp[cache_key] < self.cache_ttl):
            return self._cache[cache_key]

        # 计算新统计信息
        stats = self._calculate_statistics(manager)

        # 更新缓存
        self._cache[cache_key] = stats
        self._cache_timestamp[cache_key] = current_time

        return stats
```

## 安全审计日志的最佳实践

### 1. **敏感操作审计**
```python
# ✅ 记录所有敏感操作
sensitive_actions = [
    "privilege_change",
    "data_export",
    "user_deletion",
    "security_config_change",
    "api_key_generation"
]

def log_sensitive_operation(user_id: str, action: str, **kwargs):
    """记录敏感操作"""
    event = create_audit_event(
        user_id=user_id,
        action=action,
        resource="sensitive_operation",
        severity=AuditSeverity.HIGH,
        additional_data={
            "operation_details": kwargs,
            "requires_approval": True,
            "compliance_impact": "high"
        }
    )

    # 同步记录敏感操作
    audit_manager.log_event(event, status="success")

    # 可选：发送实时告警
    if action in ["privilege_change", "user_deletion"]:
        send_security_alert(event)
```

### 2. **失败事件处理**
```python
def log_failed_operation(user_id: str, action: str, error: Exception):
    """记录失败操作"""
    event = create_audit_event(
        user_id=user_id,
        action=action,
        resource="operation_failed",
        severity=AuditSeverity.WARNING
    )

    log_entry = audit_manager.log_event(
        event,
        status="failed",
        message=f"Operation failed: {str(error)}",
        error_details={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc()
        }
    )

    # 检查是否需要安全响应
    if _is_security_relevant_failure(action, error):
        trigger_security_response(log_entry)
```

### 3. **会话审计**
```python
class SessionAuditor:
    def __init__(self, audit_manager: AuditLogManager):
        self.audit_manager = audit_manager
        self.active_sessions = {}

    def start_session(self, user_id: str, session_id: str, context: Dict):
        """开始会话审计"""
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "context": context
        }

        # 记录会话开始
        event = create_audit_event(
            user_id=user_id,
            action="session_start",
            resource="session",
            additional_data={
                "session_id": session_id,
                "login_method": context.get("method"),
                "ip_address": context.get("ip_address")
            }
        )
        self.audit_manager.log_event(event)

    def end_session(self, session_id: str, reason: str = "logout"):
        """结束会话审计"""
        if session_id not in self.active_sessions:
            return

        session_info = self.active_sessions.pop(session_id)
        duration = datetime.now() - session_info["start_time"]

        # 记录会话结束
        event = create_audit_event(
            user_id=session_info["user_id"],
            action="session_end",
            resource="session",
            additional_data={
                "session_id": session_id,
                "duration_seconds": duration.total_seconds(),
                "end_reason": reason
            }
        )
        self.audit_manager.log_event(event)
```

## 合规性要求

### 1. **GDPR合规**
```python
class GDPRComplianceAuditor:
    """GDPR合规审计"""

    def log_data_access(self, user_id: str, data_subject_id: str, purpose: str):
        """记录数据访问（GDPR第15条）"""
        event = create_audit_event(
            user_id=user_id,
            action="data_access",
            resource="personal_data",
            severity=AuditSeverity.INFO,
            additional_data={
                "data_subject_id": data_subject_id,
                "legal_basis": "legitimate_interest",
                "purpose": purpose,
                "gdpr_article": "15"  # 数据主体访问权
            }
        )
        audit_manager.log_event(event)

    def log_consent_processing(self, user_id: str, consent_id: str, action: str):
        """记录同意处理（GDPR第7条）"""
        event = create_audit_event(
            user_id=user_id,
            action=action,
            resource="consent",
            severity=AuditSeverity.HIGH,
            additional_data={
                "consent_id": consent_id,
                "gdpr_article": "7",  # 同意条件
                "processing_purpose": "profiling"
            }
        )
        audit_manager.log_event(event)
```

### 2. **SOX合规**
```python
class SOXComplianceAuditor:
    """SOX合规审计"""

    def log_financial_access(self, user_id: str, record_type: str, record_id: str):
        """记录财务数据访问（SOX 404）"""
        event = create_audit_event(
            user_id=user_id,
            action="financial_data_access",
            resource="financial_records",
            severity=AuditSeverity.HIGH,
            additional_data={
                "record_type": record_type,
                "record_id": record_id,
                "sox_section": "404",
                "segregation_required": True
            }
        )
        audit_manager.log_event(event)

    def log_control_change(self, user_id: str, control_id: str, change_type: str):
        """记录控制变更（SOX 302）"""
        event = create_audit_event(
            user_id=user_id,
            action="internal_control_change",
            resource="financial_controls",
            severity=AuditSeverity.CRITICAL,
            additional_data={
                "control_id": control_id,
                "change_type": change_type,
                "sox_section": "302",
                "requires_cfo_review": True
            }
        )
        audit_manager.log_event(event)
```

## 监控和告警

### 1. **异常检测**
```python
class AnomalyDetector:
    def __init__(self, audit_manager: AuditLogManager):
        self.audit_manager = audit_manager
        self.baseline_metrics = {}

    def detect_login_anomalies(self):
        """检测登录异常"""
        recent_events = self.audit_manager.search_events(
            AuditSearchFilter(
                actions=["login", "failed_login"],
                start_time=datetime.now() - timedelta(hours=1)
            )
        )

        # 统计失败登录次数
        failed_logins = [e for e in recent_events if e.status == "failed"]

        # 检测暴力破解
        user_failures = defaultdict(int)
        for event in failed_logins:
            user_failures[event.event.user_id] += 1

        # 告警阈值
        for user_id, failure_count in user_failures.items():
            if failure_count >= 5:
                self.trigger_security_alert(
                    "Potential brute force attack",
                    user_id=user_id,
                    failure_count=failure_count
                )

    def detect_privilege_escalation(self):
        """检测权限提升异常"""
        recent_events = self.audit_manager.search_events(
            AuditSearchFilter(
                actions=["privilege_change", "role_assignment"],
                start_time=datetime.now() - timedelta(hours=24)
            )
        )

        # 检测异常权限变更
        for event in recent_events:
            if event.event.severity == AuditSeverity.CRITICAL:
                self.trigger_security_alert(
                    "Critical privilege escalation detected",
                    event_details=event.to_dict()
                )
```

### 2. **实时告警**
```python
class AuditAlertManager:
    def __init__(self):
        self.alert_handlers = []

    def register_alert_handler(self, handler):
        """注册告警处理器"""
        self.alert_handlers.append(handler)

    def trigger_security_alert(self, message: str, **kwargs):
        """触发安全告警"""
        alert = {
            "timestamp": datetime.now(),
            "message": message,
            "severity": kwargs.get("severity", "high"),
            "details": kwargs
        }

        # 通知所有处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

# 告警处理器示例
def email_alert_handler(alert):
    """邮件告警处理器"""
    if alert["severity"] in ["high", "critical"]:
        send_security_email(
            recipients=["security@company.com"],
            subject=f"Security Alert: {alert['message']}",
            body=json.dumps(alert["details"], indent=2)
        )

def slack_alert_handler(alert):
    """Slack告警处理器"""
    send_slack_message(
        channel="#security-alerts",
        message=f"🚨 {alert['message']}",
        attachments=[alert["details"]]
    )
```

## 性能优化技巧

### 1. **索引策略**
```python
class IndexedAuditLog:
    def __init__(self):
        self.events = []
        self.user_index = defaultdict(list)    # 用户索引
        self.time_index = []                  # 时间索引
        self.severity_index = defaultdict(list)  # 严重级别索引

    def add_event(self, event: AuditLog):
        """添加事件并更新索引"""
        self.events.append(event)

        # 更新用户索引
        self.user_index[event.event.user_id].append(len(self.events) - 1)

        # 更新时间索引（保持排序）
        bisect.insort(self.time_index, (event.timestamp, len(self.events) - 1))

        # 更新严重级别索引
        self.severity_index[event.event.severity].append(len(self.events) - 1)

    def search_by_user(self, user_id: str) -> List[AuditLog]:
        """使用用户索引快速搜索"""
        event_indices = self.user_index.get(user_id, [])
        return [self.events[i] for i in event_indices]
```

### 2. **压缩存储**
```python
import gzip
import pickle

class CompressedAuditStorage:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.current_batch = []

    def add_event(self, event: AuditLog):
        """添加事件到批次"""
        self.current_batch.append(event)

        if len(self.current_batch) >= self.batch_size:
            self.compress_and_store()

    def compress_and_store(self):
        """压缩并存储当前批次"""
        if not self.current_batch:
            return

        # 序列化
        data = pickle.dumps(self.current_batch)

        # 压缩
        compressed = gzip.compress(data)

        # 存储
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_logs_{timestamp}.gz"

        with open(filename, 'wb') as f:
            f.write(compressed)

        self.current_batch.clear()
        logger.info(f"Compressed and stored {len(self.current_batch)} events to {filename}")
```

## 故障排除指南

### 常见问题和解决方案

**Q: 审计日志占用内存过多**
```python
# A: 实施定期清理策略
def cleanup_old_logs(manager: AuditLogManager, retention_days: int = 90):
    """清理超出保留期的日志"""
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted_count = manager.clear_old_events(cutoff_time)
    logger.info(f"Cleaned up {deleted_count} old audit logs")
```

**Q: 搜索性能下降**
```python
# A: 实施搜索优化
def optimize_search_performance(manager: AuditLogManager):
    """优化搜索性能"""
    # 1. 限制搜索时间范围
    max_search_range = timedelta(days=30)

    # 2. 使用更精确的过滤条件
    # 3. 避免全表扫描
    # 4. 考虑使用外部搜索引擎
    pass
```

**Q: 事件丢失**
```python
# A: 实施持久化和重试机制
class ReliableAuditLogger:
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.retry_queue = []

    def log_event_with_retry(self, event: AuditLog, max_retries: int = 3):
        """带重试机制的事件记录"""
        for attempt in range(max_retries):
            try:
                self.storage.write_event(event)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，加入重试队列
                    self.retry_queue.append((event, time.time()))
                    logger.error(f"Failed to log event after {max_retries} attempts: {e}")
                else:
                    time.sleep(2 ** attempt)  # 指数退避
```

审计日志系统是企业级应用安全基础设施的重要组成部分，通过合理的设计和实现，可以为系统提供全面的安全监控和合规支持。