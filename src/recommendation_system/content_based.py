"""
内容推荐引擎实现

基于内容特征进行推荐的核心算法实现
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import time
from collections import defaultdict

from .models import (
    UserProfile, ContentFeature, RecommendationResult,
    RecommendationItem
)


class ContentBasedEngine:
    """内容推荐引擎"""

    def __init__(self):
        """初始化内容推荐引擎"""
        self.content_features: Dict[str, ContentFeature] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.feature_weights: Dict[str, float] = {
            "category": 0.3,  # 类别相似度权重
            "features": 0.5,  # 特征相似度权重
            "tags": 0.2       # 标签相似度权重
        }
        self.similarity_cache: Dict[Tuple[str, str], float] = {}

    def add_content_feature(self, item_id: str, category: str,
                          features: Dict[str, float], tags: List[str] = None) -> ContentFeature:
        """
        添加内容特征

        Args:
            item_id: 物品ID
            category: 内容类别
            features: 特征字典
            tags: 标签列表

        Returns:
            ContentFeature: 创建的内容特征对象
        """
        content_feature = ContentFeature(item_id, category)

        # 添加特征
        for feature_name, value in features.items():
            content_feature.add_feature(feature_name, value)

        # 添加标签
        if tags:
            for tag in tags:
                content_feature.add_tag(tag)

        self.content_features[item_id] = content_feature
        return content_feature

    def create_user_profile(self, user_id: str) -> UserProfile:
        """
        创建用户画像

        Args:
            user_id: 用户ID

        Returns:
            UserProfile: 创建的用户画像对象
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]

    def update_user_preference(self, user_id: str, category: str, score: float):
        """
        更新用户偏好

        Args:
            user_id: 用户ID
            category: 偏好类别
            score: 偏好分数 (0-1)
        """
        if score < 0 or score > 1:
            raise ValueError(f"偏好分数必须在0-1之间，当前值: {score}")

        # 自动创建用户画像（如果不存在）
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)

        self.user_profiles[user_id].update_preference(category, score)

    def update_user_behavior_feature(self, user_id: str, feature: str, value: float):
        """
        更新用户行为特征

        Args:
            user_id: 用户ID
            feature: 特征名称
            value: 特征值
        """
        # 自动创建用户画像（如果不存在）
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)

        self.user_profiles[user_id].update_behavior_feature(feature, value)

    def _calculate_content_similarity(self, features_a: Dict[str, float],
                                   features_b: Dict[str, float]) -> float:
        """
        计算内容相似度

        Args:
            features_a: 物品A的特征
            features_b: 物品B的特征

        Returns:
            float: 相似度分数 (0-1)
        """
        if not features_a or not features_b:
            return 0.0

        # 获取共同特征
        common_features = set(features_a.keys()) & set(features_b.keys())

        if not common_features:
            return 0.0

        # 计算余弦相似度
        vector_a = np.array([features_a[feature] for feature in common_features])
        vector_b = np.array([features_b[feature] for feature in common_features])

        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _calculate_category_similarity(self, category_a: str, category_b: str) -> float:
        """计算类别相似度"""
        return 1.0 if category_a == category_b else 0.0

    def _calculate_tags_similarity(self, tags_a: List[str], tags_b: List[str]) -> float:
        """计算标签相似度（Jaccard相似度）"""
        if not tags_a or not tags_b:
            return 0.0

        set_a = set(tags_a)
        set_b = set(tags_b)
        intersection = set_a & set_b
        union = set_a | set_b

        return len(intersection) / len(union) if union else 0.0

    def _calculate_overall_similarity(self, item_a: str, item_b: str) -> float:
        """
        计算整体相似度

        Args:
            item_a: 物品A的ID
            item_b: 物品B的ID

        Returns:
            float: 整体相似度分数
        """
        if item_a == item_b:
            return 1.0

        if item_a not in self.content_features or item_b not in self.content_features:
            return 0.0

        # 检查缓存
        cache_key = tuple(sorted([item_a, item_b]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        feature_a = self.content_features[item_a]
        feature_b = self.content_features[item_b]

        # 计算各部分相似度
        category_sim = self._calculate_category_similarity(
            feature_a.category, feature_b.category
        )
        features_sim = self._calculate_content_similarity(
            feature_a.features, feature_b.features
        )
        tags_sim = self._calculate_tags_similarity(
            feature_a.tags, feature_b.tags
        )

        # 加权求和
        overall_sim = (
            self.feature_weights["category"] * category_sim +
            self.feature_weights["features"] * features_sim +
            self.feature_weights["tags"] * tags_sim
        )

        # 缓存结果
        self.similarity_cache[cache_key] = overall_sim
        return overall_sim

    def find_similar_content(self, target_item: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        找到与目标物品相似的内容

        Args:
            target_item: 目标物品ID
            k: 返回的相似物品数量

        Returns:
            List[Tuple[str, float]]: 相似物品列表，格式为 (item_id, similarity_score)
        """
        if target_item not in self.content_features:
            return []

        similarities = []
        for item_id in self.content_features:
            if item_id != target_item:
                similarity = self._calculate_overall_similarity(target_item, item_id)
                similarities.append((item_id, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def generate_recommendations(self, user_id: str, k: int = 10,
                               exclude_items: Set[str] = None) -> RecommendationResult:
        """
        为用户生成推荐

        Args:
            user_id: 用户ID
            k: 推荐数量
            exclude_items: 要排除的物品集合

        Returns:
            RecommendationResult: 推荐结果
        """
        if user_id not in self.user_profiles:
            raise ValueError(f"用户 {user_id} 不存在")

        if exclude_items is None:
            exclude_items = set()

        user_profile = self.user_profiles[user_id]
        recommendations = []

        # 计算每个物品的推荐分数
        for item_id, content_feature in self.content_features.items():
            if item_id in exclude_items:
                continue

            score = self._calculate_recommendation_score(user_profile, content_feature)
            if score > 0:
                recommendations.append(RecommendationItem(item_id, score))

        # 按分数排序并返回前k个
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return RecommendationResult(user_id, "content_based", recommendations[:k])

    def _calculate_recommendation_score(self, user_profile: UserProfile,
                                     content_feature: ContentFeature) -> float:
        """
        计算推荐分数

        Args:
            user_profile: 用户画像
            content_feature: 内容特征

        Returns:
            float: 推荐分数
        """
        score = 0.0

        # 基于用户偏好
        if content_feature.category in user_profile.preferences:
            score += user_profile.preferences[content_feature.category] * 0.6

        # 基于内容特征匹配
        if content_feature.features:
            # 这里简化处理，实际中应该更复杂
            feature_match_score = 0.0
            for feature_name, value in content_feature.features.items():
                # 检查用户行为特征中是否有相关偏好
                related_behavior = user_profile.behavior_features.get(f"pref_{feature_name}", 0)
                feature_match_score += value * related_behavior * 0.3

            score += feature_match_score

        # 添加一些随机性以确保多样性
        score += np.random.random() * 0.1

        return min(score, 1.0)

    def get_popular_content(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        获取热门内容

        Args:
            k: 返回的数量

        Returns:
            List[Tuple[str, float]]: 热门内容列表，格式为 (item_id, popularity_score)
        """
        # 简化实现：基于特征数量和标签数量计算热度
        popularity_scores = []
        for item_id, content_feature in self.content_features.items():
            score = len(content_feature.features) + len(content_feature.tags)
            popularity_scores.append((item_id, float(score)))

        # 按热度排序
        popularity_scores.sort(key=lambda x: x[1], reverse=True)
        return popularity_scores[:k]

    def clear_cache(self):
        """清空缓存"""
        self.similarity_cache.clear()

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "num_users": len(self.user_profiles),
            "num_content_items": len(self.content_features),
            "cache_size": len(self.similarity_cache),
            "feature_weights": self.feature_weights.copy()
        }

    def update_feature_weights(self, new_weights: Dict[str, float]):
        """
        更新特征权重

        Args:
            new_weights: 新的权重字典

        Raises:
            ValueError: 当权重总和不等于1.0时
        """
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"特征权重总和必须等于1.0，当前值: {total_weight}")

        self.feature_weights = new_weights.copy()
        # 清空缓存，因为相似度计算方式已改变
        self.clear_cache()