# 数据模型与存储设计

## 📊 数据模型概览

本推荐系统采用多层次的数据模型设计，支持智能体社交平台的复杂推荐场景，包括用户-物品交互、社交网络关系、实时数据流处理等。

```
数据模型架构：
┌─────────────────────────────────────────────────────────┐
│                🏗️ 应用数据模型层                         │
│        • 推荐结果模型 • 用户画像模型 • 上下文模型        │
├─────────────────────────────────────────────────────────┤
│                🗄️ 业务数据模型层                         │
│        • 交互数据模型 • 社交网络模型 • 内容特征模型      │
├─────────────────────────────────────────────────────────┤
│                💾 存储数据模型层                         │
│        • 关系型数据库 • NoSQL数据库 • 缓存数据模型      │
└─────────────────────────────────────────────────────────┘
```

## 🏗️ 应用数据模型

### 1. 推荐结果模型

#### RecommendationItem 推荐项模型
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

@dataclass
class RecommendationItem:
    """推荐项数据模型"""
    item_id: str
    score: float
    reason: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'item_id': self.item_id,
            'score': self.score,
            'reason': self.reason,
            'explanation': self.explanation,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecommendationItem':
        """从字典创建对象"""
        return cls(
            item_id=data['item_id'],
            score=data['score'],
            reason=data.get('reason'),
            explanation=data.get('explanation'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        )

@dataclass
class RecommendationResult:
    """推荐结果集合模型"""
    user_id: str
    method: str
    items: List[RecommendationItem]
    request_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'user_id': self.user_id,
            'method': self.method,
            'items': [item.to_dict() for item in self.items],
            'request_id': self.request_id,
            'generated_at': self.generated_at.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'cache_hit': self.cache_hit
        }
```

#### UserInteraction 用户交互模型
```python
from enum import Enum
from typing import Union

class InteractionType(Enum):
    """交互类型枚举"""
    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    COLLABORATE = "collaborate"
    FOLLOW = "follow"
    RATE = "rate"

@dataclass
class UserInteraction:
    """用户交互数据模型"""
    user_id: str
    item_id: str
    interaction_type: InteractionType
    rating: Optional[float] = None
    context: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'interaction_type': self.interaction_type.value,
            'rating': self.rating,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'metadata': self.metadata
        }

@dataclass
class InteractionSession:
    """用户会话模型"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    interactions: List[UserInteraction] = field(default_factory=list)
    context: Optional[Dict[str, Any]] = field(default_factory=dict)

    def add_interaction(self, interaction: UserInteraction):
        """添加交互记录"""
        self.interactions.append(interaction)
        self.end_time = interaction.timestamp

    def get_duration_minutes(self) -> float:
        """获取会话时长（分钟）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0.0
```

### 2. 用户画像模型

#### UserProfile 用户画像
```python
@dataclass
class UserDemographics:
    """用户人口统计信息"""
    age_group: Optional[str] = None
    location: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    device_type: Optional[str] = None

@dataclass
class UserPreferences:
    """用户偏好信息"""
    categories: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    preferred_interaction_types: List[str] = field(default_factory=list)

@dataclass
class UserStatistics:
    """用户统计信息"""
    total_interactions: int = 0
    avg_rating: float = 0.0
    diversity_score: float = 0.0
    activity_level: str = "low"  # low, medium, high
    last_active: Optional[datetime] = None
    registration_date: Optional[datetime] = None

@dataclass
class UserProfile:
    """用户画像完整模型"""
    user_id: str
    demographics: UserDemographics = field(default_factory=UserDemographics)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    statistics: UserStatistics = field(default_factory=UserStatistics)
    feature_vector: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update_activity(self, interaction: UserInteraction):
        """更新用户活动统计"""
        self.statistics.total_interactions += 1

        if interaction.rating:
            # 更新平均评分
            total_rating = self.statistics.avg_rating * (self.statistics.total_interactions - 1)
            self.statistics.avg_rating = (total_rating + interaction.rating) / self.statistics.total_interactions

        self.statistics.last_active = interaction.timestamp
        self.updated_at = datetime.now()

        # 更新活动等级
        if self.statistics.total_interactions > 100:
            self.statistics.activity_level = "high"
        elif self.statistics.total_interactions > 20:
            self.statistics.activity_level = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'user_id': self.user_id,
            'demographics': self.demographics.__dict__,
            'preferences': self.preferences.__dict__,
            'statistics': self.statistics.__dict__,
            'feature_vector': self.feature_vector.tolist() if self.feature_vector is not None else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
```

### 3. 上下文模型

#### RecommendationContext 推荐上下文
```python
@dataclass
class DeviceContext:
    """设备上下文"""
    device_type: str  # mobile, desktop, tablet
    os: Optional[str] = None
    browser: Optional[str] = None
    screen_resolution: Optional[str] = None

@dataclass
class LocationContext:
    """位置上下文"""
    country: Optional[str] = None
    city: Optional[str] = None
    timezone: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None

@dataclass
class TimeContext:
    """时间上下文"""
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    season: str  # spring, summer, fall, winter
    is_holiday: bool = False

@dataclass
class SocialContext:
    """社交上下文"""
    friends_count: int = 0
    active_friends: List[str] = field(default_factory=list)
    current_groups: List[str] = field(default_factory=list)
    social_influence_score: float = 0.0

@dataclass
class RecommendationContext:
    """推荐上下文完整模型"""
    scene: str = "default"  # homepage, search, social, profile
    device: Optional[DeviceContext] = None
    location: Optional[LocationContext] = None
    time: Optional[TimeContext] = None
    social: Optional[SocialContext] = None
    business_rules: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request_data: Dict[str, Any]) -> 'RecommendationContext':
        """从请求数据创建上下文"""
        device_ctx = None
        if 'device' in request_data:
            device_ctx = DeviceContext(**request_data['device'])

        location_ctx = None
        if 'location' in request_data:
            location_ctx = LocationContext(**request_data['location'])

        time_ctx = None
        if 'time' in request_data:
            time_ctx = TimeContext(**request_data['time'])

        social_ctx = None
        if 'social' in request_data:
            social_ctx = SocialContext(**request_data['social'])

        return cls(
            scene=request_data.get('scene', 'default'),
            device=device_ctx,
            location=location_ctx,
            time=time_ctx,
            social=social_ctx,
            business_rules=request_data.get('business_rules', {})
        )
```

## 🗄️ 业务数据模型

### 1. 用户-物品交互数据模型

#### InteractionDataModel 交互数据模型
```python
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

class InteractionDataModel:
    """用户-物品交互数据模型"""

    def __init__(self):
        self.interactions: List[UserInteraction] = []
        self.user_item_matrix: Dict[str, Dict[str, float]] = {}
        self.item_user_matrix: Dict[str, Dict[str, float]] = {}
        self.user_stats: Dict[str, Dict[str, Any]] = {}
        self.item_stats: Dict[str, Dict[str, Any]] = {}

    def add_interaction(self, interaction: UserInteraction):
        """添加交互记录"""
        self.interactions.append(interaction)

        # 更新用户-物品矩阵
        if interaction.user_id not in self.user_item_matrix:
            self.user_item_matrix[interaction.user_id] = {}

        if interaction.interaction_type == InteractionType.RATE and interaction.rating:
            self.user_item_matrix[interaction.user_id][interaction.item_id] = interaction.rating
        elif interaction.interaction_type in [InteractionType.LIKE, InteractionType.CLICK]:
            # 将点击和喜欢转换为隐式评分
            current_rating = self.user_item_matrix[interaction.user_id].get(interaction.item_id, 0)
            self.user_item_matrix[interaction.user_id][interaction.item_id] = min(current_rating + 1.0, 5.0)

        # 更新物品-用户矩阵
        if interaction.item_id not in self.item_user_matrix:
            self.item_user_matrix[interaction.item_id] = {}

        if interaction.interaction_type == InteractionType.RATE and interaction.rating:
            self.item_user_matrix[interaction.item_id][interaction.user_id] = interaction.rating

        # 更新统计信息
        self._update_user_stats(interaction)
        self._update_item_stats(interaction)

    def _update_user_stats(self, interaction: UserInteraction):
        """更新用户统计信息"""
        user_id = interaction.user_id
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                'total_interactions': 0,
                'total_ratings': 0,
                'avg_rating': 0.0,
                'rating_variance': 0.0,
                'interaction_types': defaultdict(int),
                'last_interaction': None,
                'favorite_items': set(),
                'categories': defaultdict(float)
            }

        stats = self.user_stats[user_id]
        stats['total_interactions'] += 1
        stats['interaction_types'][interaction.interaction_type.value] += 1
        stats['last_interaction'] = interaction.timestamp

        if interaction.rating:
            stats['total_ratings'] += 1
            # 更新平均评分
            old_avg = stats['avg_rating']
            old_count = stats['total_ratings'] - 1
            new_avg = (old_avg * old_count + interaction.rating) / stats['total_ratings']
            stats['avg_rating'] = new_avg

    def _update_item_stats(self, interaction: UserInteraction):
        """更新物品统计信息"""
        item_id = interaction.item_id
        if item_id not in self.item_stats:
            self.item_stats[item_id] = {
                'total_interactions': 0,
                'total_ratings': 0,
                'avg_rating': 0.0,
                'rating_variance': 0.0,
                'popularity_score': 0.0,
                'interaction_types': defaultdict(int),
                'rating_history': [],
                'user_demographics': defaultdict(int)
            }

        stats = self.item_stats[item_id]
        stats['total_interactions'] += 1
        stats['interaction_types'][interaction.interaction_type.value] += 1

        if interaction.rating:
            stats['total_ratings'] += 1
            stats['rating_history'].append({
                'rating': interaction.rating,
                'timestamp': interaction.timestamp,
                'user_id': interaction.user_id
            })

            # 更新平均评分
            old_avg = stats['avg_rating']
            old_count = stats['total_ratings'] - 1
            new_avg = (old_avg * old_count + interaction.rating) / stats['total_ratings']
            stats['avg_rating'] = new_avg

        # 更新流行度分数
        stats['popularity_score'] = self._calculate_popularity_score(item_id)

    def _calculate_popularity_score(self, item_id: str) -> float:
        """计算物品流行度分数"""
        if item_id not in self.item_stats:
            return 0.0

        stats = self.item_stats[item_id]

        # 基础流行度：交互数量
        base_score = np.log1p(stats['total_interactions'])

        # 时间衰减：最近交互加权
        now = datetime.now()
        recent_weight = 0.0
        for interaction in self.interactions[-100:]:  # 最近100次交互
            if interaction.item_id == item_id:
                days_ago = (now - interaction.timestamp).days
                time_weight = np.exp(-days_ago / 30)  # 30天衰减周期
                recent_weight += time_weight

        return base_score * (1 + recent_weight * 0.5)

    def get_user_vector(self, user_id: str, feature_dim: int = 100) -> np.ndarray:
        """获取用户特征向量"""
        if user_id not in self.user_stats:
            return np.zeros(feature_dim)

        # 基于用户统计信息构建特征向量
        stats = self.user_stats[user_id]
        features = np.zeros(feature_dim)

        # 基础特征
        features[0] = np.log1p(stats['total_interactions'])
        features[1] = stats['avg_rating'] / 5.0  # 归一化
        features[2] = len(stats['favorite_items']) / 100.0  # 归一化

        # 交互类型特征
        interaction_types = ['view', 'click', 'like', 'share', 'collaborate']
        for i, int_type in enumerate(interaction_types):
            if int_type in stats['interaction_types']:
                features[3 + i] = stats['interaction_types'][int_type] / stats['total_interactions']

        return features

    def get_item_vector(self, item_id: str, feature_dim: int = 100) -> np.ndarray:
        """获取物品特征向量"""
        if item_id not in self.item_stats:
            return np.zeros(feature_dim)

        stats = self.item_stats[item_id]
        features = np.zeros(feature_dim)

        # 基础特征
        features[0] = np.log1p(stats['total_interactions'])
        features[1] = stats['avg_rating'] / 5.0  # 归一化
        features[2] = stats['popularity_score'] / 10.0  # 归一化

        # 交互类型分布
        interaction_types = ['view', 'click', 'like', 'share', 'collaborate']
        for i, int_type in enumerate(interaction_types):
            if int_type in stats['interaction_types']:
                features[3 + i] = stats['interaction_types'][int_type] / stats['total_interactions']

        return features
```

### 2. 社交网络数据模型

#### SocialNetworkModel 社交网络模型
```python
from typing import Set, List, Tuple, Dict
from collections import defaultdict, deque
import networkx as nx

class SocialConnection:
    """社交连接模型"""
    def __init__(self, user_a: str, user_b: str, strength: float, connection_type: str = "friend"):
        self.user_a = user_a
        self.user_b = user_b
        self.strength = strength  # 0.0 - 1.0
        self.connection_type = connection_type  # friend, follow, collaborate
        self.created_at: datetime = field(default_factory=datetime.now)
        self.interactions_count: int = 0
        self.last_interaction: Optional[datetime] = None

class SocialNetworkModel:
    """社交网络数据模型"""

    def __init__(self):
        self.connections: Dict[str, List[SocialConnection]] = defaultdict(list)
        self.user_influence: Dict[str, float] = {}
        self.trust_scores: Dict[Tuple[str, str], float] = {}
        self.community_assignments: Dict[str, str] = {}
        self.graph = nx.Graph()

    def add_connection(self, connection: SocialConnection):
        """添加社交连接"""
        # 双向添加连接
        self.connections[connection.user_a].append(connection)
        self.connections[connection.user_b].append(connection)

        # 更新图结构
        self.graph.add_edge(connection.user_a, connection.user_b,
                          weight=connection.strength,
                          type=connection.connection_type)

        # 清除相关缓存
        self._clear_cache_for_users([connection.user_a, connection.user_b])

    def get_friends(self, user_id: str, connection_type: str = None) -> List[str]:
        """获取用户的朋友列表"""
        friends = []
        if user_id in self.connections:
            for connection in self.connections[user_id]:
                if connection_type is None or connection.connection_type == connection_type:
                    friend_id = connection.user_b if connection.user_a == user_id else connection.user_a
                    friends.append((friend_id, connection.strength))

        # 按连接强度排序
        friends.sort(key=lambda x: x[1], reverse=True)
        return [friend_id for friend_id, _ in friends]

    def calculate_social_influence(self, source_user: str, target_user: str, max_depth: int = 3) -> float:
        """计算社交影响力"""
        if source_user == target_user:
            return 1.0

        if source_user not in self.connections:
            return 0.0

        # 使用BFS搜索影响力路径
        visited = set()
        queue = deque([(source_user, 1.0, 0)])  # (user, influence, depth)

        while queue:
            current_user, current_influence, depth = queue.popleft()

            if current_user in visited or depth >= max_depth:
                continue

            visited.add(current_user)

            if current_user == target_user:
                return current_influence

            if current_user in self.connections:
                user_influence = self.user_influence.get(current_user, 0.5)

                for connection in self.connections[current_user]:
                    friend_id = connection.user_b if connection.user_a == current_user else connection.user_a

                    if friend_id not in visited:
                        # 影响力衰减
                        new_influence = (current_influence *
                                      connection.strength *
                                      user_influence *
                                      (0.8 ** depth))  # 距离衰减
                        queue.append((friend_id, new_influence, depth + 1))

        return 0.0

    def detect_communities(self) -> Dict[str, str]:
        """检测社交社区"""
        try:
            import community as community_louvain

            # 使用Louvain算法检测社区
            communities = community_louvain.best_partition(self.graph)
            self.community_assignments = communities
            return communities

        except ImportError:
            # 如果没有安装community库，使用简单的连通分量
            communities = {}
            component_id = 0

            for component in nx.connected_components(self.graph):
                for user_id in component:
                    communities[user_id] = f"component_{component_id}"
                component_id += 1

            self.community_assignments = communities
            return communities

    def get_user_network_features(self, user_id: str) -> Dict[str, float]:
        """获取用户网络特征"""
        features = {
            'degree_centrality': 0.0,
            'betweenness_centrality': 0.0,
            'closeness_centrality': 0.0,
            'clustering_coefficient': 0.0,
            'influence_score': 0.0,
            'friends_count': 0,
            'community_size': 0
        }

        if user_id in self.graph:
            # 度中心性
            features['degree_centrality'] = nx.degree_centrality(self.graph)[user_id]

            # 介数中心性
            features['betweenness_centrality'] = nx.betweenness_centrality(self.graph)[user_id]

            # 接近中心性
            features['closeness_centrality'] = nx.closeness_centrality(self.graph)[user_id]

            # 聚类系数
            features['clustering_coefficient'] = nx.clustering(self.graph)[user_id]

            # 影响力分数
            features['influence_score'] = self.user_influence.get(user_id, 0.0)

            # 朋友数量
            features['friends_count'] = len(list(self.graph.neighbors(user_id)))

            # 社区大小
            if user_id in self.community_assignments:
                community_id = self.community_assignments[user_id]
                community_members = [uid for uid, cid in self.community_assignments.items() if cid == community_id]
                features['community_size'] = len(community_members)

        return features

    def _clear_cache_for_users(self, user_ids: List[str]):
        """清除用户相关缓存"""
        for user_id in user_ids:
            keys_to_remove = [key for key in self.trust_scores.keys() if user_id in key]
            for key in keys_to_remove:
                del self.trust_scores[key]
```

### 3. 内容特征模型

#### ContentFeatureModel 内容特征模型
```python
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class ContentFeatures:
    """内容特征模型"""

    def __init__(self, item_id: str):
        self.item_id = item_id
        self.text_features: Dict[str, float] = {}
        self.category_features: Dict[str, float] = {}
        self.numeric_features: Dict[str, float] = {}
        self.embedding_features: np.ndarray = None
        self.feature_vector: np.ndarray = None
        self.tags: List[str] = []
        self.created_at: datetime = field(default_factory=datetime.now)
        self.updated_at: datetime = field(default_factory=datetime.now)

class ContentFeatureModel:
    """内容特征模型管理器"""

    def __init__(self, feature_dim: int = 100):
        self.feature_dim = feature_dim
        self.items: Dict[str, ContentFeatures] = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.is_fitted = False

    def add_item(self, item_data: Dict[str, Any]) -> ContentFeatures:
        """添加物品特征"""
        item_id = item_data['item_id']
        features = ContentFeatures(item_id)

        # 文本特征
        if 'description' in item_data:
            features.text_features = self._extract_text_features(item_data['description'])

        if 'title' in item_data:
            title_features = self._extract_text_features(item_data['title'])
            for key, value in title_features.items():
                features.text_features[f"title_{key}"] = value

        # 类别特征
        if 'categories' in item_data:
            features.category_features = self._encode_categories(item_data['categories'])

        # 数值特征
        if 'numeric_features' in item_data:
            features.numeric_features = item_data['numeric_features']

        # 标签
        if 'tags' in item_data:
            features.tags = item_data['tags']

        # 生成综合特征向量
        features.feature_vector = self._create_feature_vector(features)

        self.items[item_id] = features
        return features

    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """提取文本特征"""
        if not self.is_fitted:
            # 如果还没有训练，使用简单的词频
            words = text.lower().split()
            features = {}
            for word in words:
                if len(word) > 2:  # 忽略短词
                    features[f"word_{word}"] = features.get(f"word_{word}", 0) + 1
            return features
        else:
            # 使用训练好的TF-IDF向量化器
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()

            features = {}
            for i, feature_name in enumerate(feature_names):
                if tfidf_matrix[0, i] > 0:
                    features[f"tfidf_{feature_name}"] = tfidf_matrix[0, i]

            return features

    def _encode_categories(self, categories: List[str]) -> Dict[str, float]:
        """编码类别特征"""
        features = {}
        for category in categories:
            features[f"category_{category}"] = 1.0

        return features

    def _create_feature_vector(self, features: ContentFeatures) -> np.ndarray:
        """创建综合特征向量"""
        vector = np.zeros(self.feature_dim)
        current_idx = 0

        # 文本特征
        text_features = list(features.text_features.values())
        if text_features:
            end_idx = min(current_idx + len(text_features), self.feature_dim)
            vector[current_idx:end_idx] = text_features[:end_idx - current_idx]
            current_idx = end_idx

        # 类别特征
        category_features = list(features.category_features.values())
        if category_features and current_idx < self.feature_dim:
            end_idx = min(current_idx + len(category_features), self.feature_dim)
            vector[current_idx:end_idx] = category_features[:end_idx - current_idx]
            current_idx = end_idx

        # 数值特征
        numeric_features = list(features.numeric_features.values())
        if numeric_features and current_idx < self.feature_dim:
            end_idx = min(current_idx + len(numeric_features), self.feature_dim)
            vector[current_idx:end_idx] = numeric_features[:end_idx - current_idx]

        return vector

    def train_text_features(self, documents: List[str]):
        """训练文本特征提取器"""
        if documents:
            self.tfidf_vectorizer.fit(documents)
            self.is_fitted = True

    def get_similar_items(self, item_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """获取相似物品"""
        if item_id not in self.items:
            return []

        target_vector = self.items[item_id].feature_vector
        similarities = []

        for other_id, other_features in self.items.items():
            if other_id != item_id and other_features.feature_vector is not None:
                # 计算余弦相似度
                similarity = np.dot(target_vector, other_features.feature_vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(other_features.feature_vector)
                )
                similarities.append((other_id, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def get_item_by_category(self, category: str) -> List[str]:
        """根据类别获取物品"""
        items = []
        category_key = f"category_{category}"

        for item_id, features in self.items.items():
            if category_key in features.category_features:
                items.append(item_id)

        return items
```

## 💾 存储数据模型

### 1. 关系型数据库模型

#### DatabaseSchema 数据库模式
```sql
-- 用户表
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    age_group VARCHAR(20),
    location VARCHAR(100),
    language VARCHAR(10),
    timezone VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_age_group (age_group),
    INDEX idx_users_location (location),
    INDEX idx_users_created_at (created_at)
);

-- 物品表
CREATE TABLE items (
    item_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    tags JSON,
    numeric_features JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_items_category (category),
    INDEX idx_items_created_at (created_at),
    FULLTEXT idx_items_search (title, description)
);

-- 用户交互表
CREATE TABLE user_interactions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    item_id VARCHAR(50) NOT NULL,
    interaction_type ENUM('view', 'click', 'like', 'share', 'comment', 'collaborate', 'follow', 'rate') NOT NULL,
    rating DECIMAL(3,2),
    context JSON,
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (item_id) REFERENCES items(item_id),
    INDEX idx_interactions_user_id (user_id),
    INDEX idx_interactions_item_id (item_id),
    INDEX idx_interactions_user_item (user_id, item_id),
    INDEX idx_interactions_type (interaction_type),
    INDEX idx_interactions_created_at (created_at),
    INDEX idx_interactions_user_created (user_id, created_at),
    INDEX idx_interactions_rating (rating) WHERE rating IS NOT NULL
);

-- 社交关系表
CREATE TABLE social_connections (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_a VARCHAR(50) NOT NULL,
    user_b VARCHAR(50) NOT NULL,
    connection_type ENUM('friend', 'follow', 'collaborate') NOT NULL,
    strength DECIMAL(3,2) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    interactions_count INT DEFAULT 0,
    last_interaction TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_a) REFERENCES users(user_id),
    FOREIGN KEY (user_b) REFERENCES users(user_id),
    UNIQUE KEY unique_connection (user_a, user_b, connection_type),
    INDEX idx_connections_user_a (user_a),
    INDEX idx_connections_user_b (user_b),
    INDEX idx_connections_strength (strength),
    INDEX idx_connections_type (connection_type)
);

-- 推荐记录表
CREATE TABLE recommendation_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    request_id VARCHAR(100),
    method VARCHAR(50) NOT NULL,
    recommendations JSON NOT NULL,
    context JSON,
    processing_time_ms DECIMAL(8,2),
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_recommendations_user_id (user_id),
    INDEX idx_recommendations_method (method),
    INDEX idx_recommendations_created_at (created_at),
    INDEX idx_recommendations_cache_hit (cache_hit)
);

-- 用户画像表
CREATE TABLE user_profiles (
    user_id VARCHAR(50) PRIMARY KEY,
    demographics JSON,
    preferences JSON,
    statistics JSON,
    feature_vector JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

#### DatabaseOperations 数据库操作
```python
import asyncpg
from typing import List, Dict, Any, Optional
import json
import asyncio

class RecommendationDatabase:
    """推荐系统数据库操作"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        """初始化连接池"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()

    async def get_user_interactions(self, user_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取用户交互记录"""
        async with self.pool.acquire() as conn:
            query = """
                SELECT user_id, item_id, interaction_type, rating, context, session_id, created_at
                FROM user_interactions
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await conn.fetch(query, user_id, limit)
            return [dict(row) for row in rows]

    async def get_item_interactions(self, item_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取物品交互记录"""
        async with self.pool.acquire() as conn:
            query = """
                SELECT user_id, item_id, interaction_type, rating, context, created_at
                FROM user_interactions
                WHERE item_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await conn.fetch(query, item_id, limit)
            return [dict(row) for row in rows]

    async def save_recommendation_log(self, log_data: Dict[str, Any]) -> str:
        """保存推荐日志"""
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO recommendation_logs
                (user_id, request_id, method, recommendations, context, processing_time_ms, cache_hit)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """
            row_id = await conn.fetchval(
                query,
                log_data['user_id'],
                log_data.get('request_id'),
                log_data['method'],
                json.dumps(log_data['recommendations']),
                json.dumps(log_data.get('context', {})),
                log_data.get('processing_time_ms'),
                log_data.get('cache_hit', False)
            )
            return str(row_id)

    async def get_user_social_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户社交连接"""
        async with self.pool.acquire() as conn:
            query = """
                SELECT user_a, user_b, connection_type, strength, interactions_count, last_interaction
                FROM social_connections
                WHERE user_a = $1 OR user_b = $1
            """
            rows = await conn.fetch(query, user_id)
            connections = []

            for row in rows:
                conn_data = dict(row)
                # 确保返回的朋友不是自己
                friend_id = conn_data['user_b'] if conn_data['user_a'] == user_id else conn_data['user_a']
                conn_data['friend_id'] = friend_id
                connections.append(conn_data)

            return connections

    async def batch_get_user_profiles(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取用户画像"""
        if not user_ids:
            return {}

        async with self.pool.acquire() as conn:
            query = """
                SELECT user_id, demographics, preferences, statistics, feature_vector
                FROM user_profiles
                WHERE user_id = ANY($1)
            """
            rows = await conn.fetch(query, user_ids)

            profiles = {}
            for row in rows:
                profile_data = dict(row)
                profiles[profile_data['user_id']] = profile_data

            return profiles

    async def update_user_statistics(self, user_id: str, stats: Dict[str, Any]):
        """更新用户统计信息"""
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO user_profiles (user_id, statistics)
                VALUES ($1, $2)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    statistics = jsonb_set(
                        jsonb_set(user_profiles.statistics, '{}', $2::jsonb),
                        '{updated_at}',
                        to_jsonb(NOW())
                    ),
                    updated_at = NOW()
            """
            await conn.execute(query, user_id, json.dumps(stats))
```

### 2. NoSQL数据模型

#### MongoDB模式设计
```python
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
from bson import ObjectId
import asyncio

class MongoRecommendationStore:
    """MongoDB推荐数据存储"""

    def __init__(self, connection_string: str, database_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.collections = {
            'user_features': self.db.user_features,
            'item_features': self.db.item_features,
            'recommendation_cache': self.db.recommendation_cache,
            'real_time_events': self.db.real_time_events,
            'feature_vectors': self.db.feature_vectors
        }

    async def save_user_feature_vector(self, user_id: str, vector: np.ndarray, metadata: Dict[str, Any] = None):
        """保存用户特征向量"""
        document = {
            '_id': user_id,
            'vector': vector.tolist(),
            'dimension': len(vector),
            'metadata': metadata or {},
            'updated_at': datetime.now()
        }

        await self.collections['user_features'].replace_one(
            {'_id': user_id},
            document,
            upsert=True
        )

    async def get_user_feature_vector(self, user_id: str) -> Optional[np.ndarray]:
        """获取用户特征向量"""
        document = await self.collections['user_features'].find_one({'_id': user_id})

        if document and 'vector' in document:
            return np.array(document['vector'])

        return None

    async def batch_get_feature_vectors(self, ids: List[str], collection_name: str) -> Dict[str, np.ndarray]:
        """批量获取特征向量"""
        collection = self.collections[collection_name]

        cursor = collection.find({'_id': {'$in': ids}}, {'vector': 1})
        documents = await cursor.to_list(length=len(ids))

        vectors = {}
        for doc in documents:
            vectors[doc['_id']] = np.array(doc['vector'])

        return vectors

    async def cache_recommendation(self, cache_key: str, recommendations: List[Dict[str, Any]], ttl_seconds: int = 3600):
        """缓存推荐结果"""
        document = {
            '_id': cache_key,
            'recommendations': recommendations,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl_seconds)
        }

        await self.collections['recommendation_cache'].replace_one(
            {'_id': cache_key},
            document,
            upsert=True
        )

        # 创建TTL索引（如果不存在）
        await self.collections['recommendation_cache'].create_index(
            'expires_at',
            expireAfterSeconds=0
        )

    async def get_cached_recommendation(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的推荐结果"""
        document = await self.collections['recommendation_cache'].find_one({'_id': cache_key})

        if document:
            return document.get('recommendations')

        return None

    async def store_real_time_event(self, event_data: Dict[str, Any]):
        """存储实时事件"""
        event_data['_id'] = ObjectId()
        event_data['timestamp'] = datetime.now()

        await self.collections['real_time_events'].insert_one(event_data)

    async def get_recent_events(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的实时事件"""
        cursor = self.collections['real_time_events'].find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(limit)

        return await cursor.to_list(length=limit)
```

### 3. 缓存数据模型

#### Redis缓存模型
```python
import aioredis
import pickle
import json
from typing import List, Dict, Any, Optional
from datetime import timedelta

class RedisCacheManager:
    """Redis缓存管理器"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        """连接Redis"""
        self.redis = await aioredis.from_url(self.redis_url)

    async def disconnect(self):
        """断开Redis连接"""
        if self.redis:
            await self.redis.close()

    async def cache_user_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]], ttl_hours: int = 1):
        """缓存用户推荐"""
        cache_key = f"user_rec:{user_id}"
        serialized_data = json.dumps(recommendations)

        await self.redis.setex(cache_key, ttl_hours * 3600, serialized_data)

    async def get_cached_user_recommendations(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的用户推荐"""
        cache_key = f"user_rec:{user_id}"
        cached_data = await self.redis.get(cache_key)

        if cached_data:
            return json.loads(cached_data)

        return None

    async def cache_feature_vector(self, entity_id: str, vector: np.ndarray, ttl_hours: int = 24):
        """缓存特征向量"""
        cache_key = f"feature_vec:{entity_id}"
        serialized_vector = pickle.dumps(vector)

        await self.redis.setex(cache_key, ttl_hours * 3600, serialized_vector)

    async def get_cached_feature_vector(self, entity_id: str) -> Optional[np.ndarray]:
        """获取缓存的特征向量"""
        cache_key = f"feature_vec:{entity_id}"
        cached_vector = await self.redis.get(cache_key)

        if cached_vector:
            return pickle.loads(cached_vector)

        return None

    async def cache_user_stats(self, user_id: str, stats: Dict[str, Any], ttl_hours: int = 6):
        """缓存用户统计信息"""
        cache_key = f"user_stats:{user_id}"
        serialized_stats = json.dumps(stats)

        await self.redis.setex(cache_key, ttl_hours * 3600, serialized_stats)

    async def get_cached_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的用户统计信息"""
        cache_key = f"user_stats:{user_id}"
        cached_stats = await self.redis.get(cache_key)

        if cached_stats:
            return json.loads(cached_stats)

        return None

    async def cache_similarity_matrix(self, matrix_key: str, similarity_data: Dict[str, float], ttl_hours: int = 12):
        """缓存相似度矩阵"""
        cache_key = f"sim_matrix:{matrix_key}"
        serialized_data = json.dumps(similarity_data)

        await self.redis.setex(cache_key, ttl_hours * 3600, serialized_data)

    async def get_cached_similarity_matrix(self, matrix_key: str) -> Optional[Dict[str, float]]:
        """获取缓存的相似度矩阵"""
        cache_key = f"sim_matrix:{matrix_key}"
        cached_data = await self.redis.get(cache_key)

        if cached_data:
            return json.loads(cached_data)

        return None

    async def invalidate_user_cache(self, user_id: str):
        """使用户缓存失效"""
        patterns = [
            f"user_rec:{user_id}",
            f"user_stats:{user_id}",
            f"feature_vec:user:{user_id}"
        ]

        for pattern in patterns:
            await self.redis.delete(pattern)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        info = await self.redis.info()

        return {
            'used_memory': info.get('used_memory_human', 'N/A'),
            'connected_clients': info.get('connected_clients', 0),
            'total_commands_processed': info.get('total_commands_processed', 0),
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
        }
```

## 🚀 实时数据流处理

### 实时数据流架构
```python
import asyncio
from typing import Callable, Dict, Any
from datetime import datetime
import json

class RealTimeDataProcessor:
    """实时数据处理器"""

    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.processing = False

    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """发布事件"""
        event = {
            'type': event_type,
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        }

        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            print(f"Event queue is full, dropping event: {event_type}")

    async def start_processing(self):
        """开始处理事件"""
        if self.processing:
            return

        self.processing = True

        while self.processing:
            try:
                # 等待事件，超时1秒
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                # 超时继续循环
                continue
            except Exception as e:
                print(f"Error processing event: {e}")

    async def _process_event(self, event: Dict[str, Any]):
        """处理单个事件"""
        event_type = event['type']
        event_data = event['data']

        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    print(f"Error in event handler for {event_type}: {e}")

    def stop_processing(self):
        """停止处理事件"""
        self.processing = False

class RecommendationEventHandlers:
    """推荐系统事件处理器"""

    def __init__(self, cache_manager: RedisCacheManager, database: RecommendationDatabase):
        self.cache_manager = cache_manager
        self.database = database

    async def handle_user_interaction(self, event_data: Dict[str, Any]):
        """处理用户交互事件"""
        user_id = event_data['user_id']
        item_id = event_data['item_id']
        interaction_type = event_data['interaction_type']

        # 使相关缓存失效
        await self.cache_manager.invalidate_user_cache(user_id)

        # 更新用户统计信息
        if interaction_type in ['like', 'rate', 'click']:
            await self._update_user_activity_stats(user_id, event_data)

        # 触发实时推荐更新（如果需要）
        if interaction_type in ['like', 'collaborate']:
            await self._trigger_recommendation_update(user_id)

    async def handle_user_profile_update(self, event_data: Dict[str, Any]):
        """处理用户画像更新事件"""
        user_id = event_data['user_id']

        # 使用户相关缓存失效
        await self.cache_manager.invalidate_user_cache(user_id)

        # 更新特征向量缓存
        if 'feature_vector' in event_data:
            vector = np.array(event_data['feature_vector'])
            await self.cache_manager.cache_feature_vector(f"user:{user_id}", vector)

    async def handle_item_update(self, event_data: Dict[str, Any]):
        """处理物品更新事件"""
        item_id = event_data['item_id']

        # 更新物品特征缓存
        if 'features' in event_data:
            vector = np.array(event_data['features'])
            await self.cache_manager.cache_feature_vector(f"item:{item_id}", vector)

        # 可能需要重新计算受影响用户的推荐
        await self._invalidate_affected_users(item_id)

    async def _update_user_activity_stats(self, user_id: str, event_data: Dict[str, Any]):
        """更新用户活动统计"""
        # 这里可以实现具体的统计更新逻辑
        pass

    async def _trigger_recommendation_update(self, user_id: str):
        """触发推荐更新"""
        # 可以发布一个推荐更新事件
        pass

    async def _invalidate_affected_users(self, item_id: str):
        """使受影响的用户缓存失效"""
        # 获取与该物品交互过的用户
        interactions = await self.database.get_item_interactions(item_id, limit=1000)

        for interaction in interactions:
            await self.cache_manager.invalidate_user_cache(interaction['user_id'])
```

这套完整的数据模型设计为百万级智能体推荐系统提供了坚实的数据基础，支持复杂的推荐场景和实时的数据处理需求。通过多层次的模型设计和多样化的存储策略，确保了系统的高性能、高可用性和良好的扩展性。