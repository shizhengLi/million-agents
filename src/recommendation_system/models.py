"""
推荐系统的数据模型定义
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from datetime import datetime


@dataclass
class RecommendationItem:
    """推荐物品项"""
    item_id: str
    score: float
    reason: Optional[str] = None

    def __post_init__(self):
        if self.score < 0 or self.score > 1:
            raise ValueError(f"推荐分数必须在0-1之间，当前值: {self.score}")


@dataclass
class RecommendationResult:
    """推荐结果"""
    user_id: str
    method: str
    items: List[RecommendationItem]
    generated_at: datetime

    def __init__(self, user_id: str, method: str, items: List[RecommendationItem]):
        self.user_id = user_id
        self.method = method
        self.items = items
        self.generated_at = datetime.now()

    def get_sorted_items(self) -> List[RecommendationItem]:
        """获取按分数排序的推荐项"""
        return sorted(self.items, key=lambda x: x.score, reverse=True)

    def get_top_k(self, k: int) -> List[RecommendationItem]:
        """获取前k个推荐项"""
        return self.get_sorted_items()[:k]


class UserItemMatrix:
    """用户-物品矩阵"""

    def __init__(self, data: np.ndarray, user_ids: List[str], item_ids: List[str]):
        """
        初始化用户-物品矩阵

        Args:
            data: 评分矩阵数据
            user_ids: 用户ID列表
            item_ids: 物品ID列表
        """
        if data.shape != (len(user_ids), len(item_ids)):
            raise ValueError("矩阵维度与用户/物品数量不匹配")

        self.data = data
        self.user_ids = user_ids
        self.item_ids = item_ids
        self._user_index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self._item_index_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

    @property
    def shape(self) -> Tuple[int, int]:
        """矩阵形状 (用户数, 物品数)"""
        return self.data.shape

    def get_user_index(self, user_id: str) -> int:
        """获取用户索引"""
        if user_id not in self._user_index_map:
            raise ValueError(f"用户 {user_id} 不存在")
        return self._user_index_map[user_id]

    def get_item_index(self, item_id: str) -> int:
        """获取物品索引"""
        if item_id not in self._item_index_map:
            raise ValueError(f"物品 {item_id} 不存在")
        return self._item_index_map[item_id]

    def get_user_vector(self, user_id: str) -> np.ndarray:
        """获取用户的评分向量"""
        user_idx = self.get_user_index(user_id)
        return self.data[user_idx, :]

    def get_item_vector(self, item_id: str) -> np.ndarray:
        """获取物品的评分向量"""
        item_idx = self.get_item_index(item_id)
        return self.data[:, item_idx]

    def get_rating(self, user_id: str, item_id: str) -> float:
        """获取用户对物品的评分"""
        user_idx = self.get_user_index(user_id)
        item_idx = self.get_item_index(item_id)
        return self.data[user_idx, item_idx]

    def set_rating(self, user_id: str, item_id: str, rating: float):
        """设置用户对物品的评分"""
        if rating < 1.0 or rating > 5.0:
            raise ValueError(f"评分必须在1.0-5.0范围内，当前值: {rating}")

        user_idx = self.get_user_index(user_id)
        item_idx = self.get_item_index(item_id)
        self.data[user_idx, item_idx] = rating

    def get_user_rated_items(self, user_id: str) -> List[str]:
        """获取用户已评分的物品列表"""
        user_idx = self.get_user_index(user_id)
        user_vector = self.data[user_idx, :]
        rated_indices = np.where(user_vector > 0)[0]
        return [self.item_ids[idx] for idx in rated_indices]

    def get_item_raters(self, item_id: str) -> List[str]:
        """获取评分过该物品的用户列表"""
        item_idx = self.get_item_index(item_id)
        item_vector = self.data[:, item_idx]
        rater_indices = np.where(item_vector > 0)[0]
        return [self.user_ids[idx] for idx in rater_indices]

    def get_common_items(self, user_a: str, user_b: str) -> List[str]:
        """获取两个用户共同评分的物品"""
        user_a_items = set(self.get_user_rated_items(user_a))
        user_b_items = set(self.get_user_rated_items(user_b))
        return list(user_a_items & user_b_items)

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        user_vector = self.get_user_vector(user_id)
        rated_items = user_vector[user_vector > 0]

        if len(rated_items) == 0:
            return {
                "rated_count": 0,
                "avg_rating": 0.0,
                "min_rating": 0.0,
                "max_rating": 0.0
            }

        return {
            "rated_count": len(rated_items),
            "avg_rating": float(np.mean(rated_items)),
            "min_rating": float(np.min(rated_items)),
            "max_rating": float(np.max(rated_items))
        }


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    demographics: Dict[str, Any]
    preferences: Dict[str, float]
    behavior_features: Dict[str, float]
    created_at: datetime
    updated_at: datetime

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.demographics = {}
        self.preferences = {}
        self.behavior_features = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def update_preference(self, category: str, score: float):
        """更新偏好分数"""
        if score < 0 or score > 1:
            raise ValueError(f"偏好分数必须在0-1之间，当前值: {score}")

        self.preferences[category] = score
        self.updated_at = datetime.now()

    def update_behavior_feature(self, feature: str, value: float):
        """更新行为特征"""
        self.behavior_features[feature] = value
        self.updated_at = datetime.now()


@dataclass
class ContentFeature:
    """内容特征"""
    item_id: str
    features: Dict[str, float]
    category: str
    tags: List[str]
    created_at: datetime

    def __init__(self, item_id: str, category: str):
        self.item_id = item_id
        self.features = {}
        self.category = category
        self.tags = []
        self.created_at = datetime.now()

    def add_feature(self, feature_name: str, value: float):
        """添加特征"""
        if value < 0:
            raise ValueError(f"特征值不能为负数，当前值: {value}")

        self.features[feature_name] = value

    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)

    def get_feature_vector(self, feature_names: List[str]) -> np.ndarray:
        """获取指定特征的向量"""
        return np.array([self.features.get(name, 0.0) for name in feature_names])


@dataclass
class Interaction:
    """用户交互记录"""
    user_id: str
    item_id: str
    rating: float
    timestamp: datetime
    interaction_type: str  # rating, view, like, share, etc.
    context: Dict[str, Any]

    def __init__(self, user_id: str, item_id: str, rating: float,
                 interaction_type: str = "rating"):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timestamp = datetime.now()
        self.interaction_type = interaction_type
        self.context = {}

        if rating < 1.0 or rating > 5.0:
            raise ValueError(f"评分必须在1.0-5.0范围内，当前值: {rating}")


@dataclass
class SimilarityResult:
    """相似度计算结果"""
    item_a: str
    item_b: str
    similarity: float
    method: str
    calculated_at: datetime

    def __init__(self, item_a: str, item_b: str, similarity: float, method: str):
        self.item_a = item_a
        self.item_b = item_b
        self.similarity = similarity
        self.method = method
        self.calculated_at = datetime.now()

        if similarity < -1 or similarity > 1:
            raise ValueError(f"相似度必须在-1到1之间，当前值: {similarity}")


@dataclass
class RecommendationMetrics:
    """推荐系统性能指标"""
    precision: float
    recall: float
    f1_score: float
    coverage: float
    diversity: float
    novelty: float
    calculated_at: datetime

    def __init__(self, precision: float, recall: float, coverage: float,
                 diversity: float, novelty: float):
        self.precision = precision
        self.recall = recall
        self.f1_score = self._calculate_f1(precision, recall)
        self.coverage = coverage
        self.diversity = diversity
        self.novelty = novelty
        self.calculated_at = datetime.now()

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)