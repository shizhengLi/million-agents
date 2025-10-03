"""
内容推荐引擎的测试用例

使用TDD方法，先写测试用例，再实现功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from recommendation_system.content_based import ContentBasedEngine
from recommendation_system.models import (
    UserProfile, ContentFeature, RecommendationResult,
    RecommendationItem
)


class TestContentBasedEngine:
    """内容推荐引擎测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.content_engine = ContentBasedEngine()

    def test_initialization(self):
        """测试内容推荐引擎初始化"""
        # Given & When
        engine = ContentBasedEngine()

        # Then
        assert engine is not None
        assert hasattr(engine, 'content_features')
        assert hasattr(engine, 'user_profiles')
        assert hasattr(engine, 'feature_weights')
        assert hasattr(engine, 'similarity_cache')

    def test_add_content_feature(self):
        """测试添加内容特征"""
        # Given
        item_id = "article_1"
        category = "technology"
        features = {
            "ai": 0.8,
            "machine_learning": 0.9,
            "python": 0.7
        }
        tags = ["AI", "Python", "Tech"]

        # When
        content_feature = self.content_engine.add_content_feature(
            item_id, category, features, tags
        )

        # Then
        assert isinstance(content_feature, ContentFeature)
        assert content_feature.item_id == item_id
        assert content_feature.category == category
        assert content_feature.features == features
        assert content_feature.tags == tags
        assert item_id in self.content_engine.content_features

    def test_add_content_feature_duplicate(self):
        """测试添加重复内容特征 - 应该更新"""
        # Given
        item_id = "article_1"
        self.content_engine.add_content_feature(item_id, "tech", {"ai": 0.8})

        # When
        updated_feature = self.content_engine.add_content_feature(
            item_id, "technology", {"ai": 0.9, "python": 0.7}
        )

        # Then - 应该更新已存在的特征
        assert updated_feature.category == "technology"
        assert updated_feature.features["ai"] == 0.9
        assert len(self.content_engine.content_features) == 1  # 仍然只有1个

    def test_create_user_profile(self):
        """测试创建用户画像"""
        # Given
        user_id = "user_1"

        # When
        user_profile = self.content_engine.create_user_profile(user_id)

        # Then
        assert isinstance(user_profile, UserProfile)
        assert user_profile.user_id == user_id
        assert user_id in self.content_engine.user_profiles

    def test_update_user_preference(self):
        """测试更新用户偏好"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)

        # When
        self.content_engine.update_user_preference(user_id, "technology", 0.8)
        self.content_engine.update_user_preference(user_id, "sports", 0.3)

        # Then
        profile = self.content_engine.user_profiles[user_id]
        assert profile.preferences["technology"] == 0.8
        assert profile.preferences["sports"] == 0.3

    def test_update_user_preference_invalid_score(self):
        """测试更新用户偏好 - 无效分数"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)

        # When & Then
        with pytest.raises(ValueError, match="偏好分数必须在0-1之间"):
            self.content_engine.update_user_preference(user_id, "tech", 1.5)

        with pytest.raises(ValueError, match="偏好分数必须在0-1之间"):
            self.content_engine.update_user_preference(user_id, "tech", -0.1)

    def test_update_user_behavior_feature(self):
        """测试更新用户行为特征"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)

        # When
        self.content_engine.update_user_behavior_feature(user_id, "view_count", 100)
        self.content_engine.update_user_behavior_feature(user_id, "like_rate", 0.7)

        # Then
        profile = self.content_engine.user_profiles[user_id]
        assert profile.behavior_features["view_count"] == 100
        assert profile.behavior_features["like_rate"] == 0.7

    def test_calculate_content_similarity(self):
        """测试计算内容相似度"""
        # Given
        feature_a = {"ai": 0.8, "python": 0.7, "database": 0.3}
        feature_b = {"ai": 0.9, "python": 0.6, "web": 0.4}

        # When
        similarity = self.content_engine._calculate_content_similarity(feature_a, feature_b)

        # Then
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # 应该有一定相似度

    def test_calculate_content_similarity_no_overlap(self):
        """测试计算内容相似度 - 无重叠特征"""
        # Given
        feature_a = {"ai": 0.8, "python": 0.7}
        feature_b = {"sports": 0.9, "music": 0.6}

        # When
        similarity = self.content_engine._calculate_content_similarity(feature_a, feature_b)

        # Then
        assert similarity == 0.0

    def test_calculate_content_similarity_identical(self):
        """测试计算内容相似度 - 相同特征"""
        # Given
        feature_a = {"ai": 0.8, "python": 0.7}
        feature_b = {"ai": 0.8, "python": 0.7}

        # When
        similarity = self.content_engine._calculate_content_similarity(feature_a, feature_b)

        # Then
        assert abs(similarity - 1.0) < 1e-6

    def test_find_similar_content(self):
        """测试查找相似内容"""
        # Given
        self.content_engine.add_content_feature("item1", "tech", {"ai": 0.8, "python": 0.7})
        self.content_engine.add_content_feature("item2", "tech", {"ai": 0.9, "python": 0.6})
        self.content_engine.add_content_feature("item3", "sports", {"football": 0.8})

        # When
        similar_items = self.content_engine.find_similar_content("item1", k=2)

        # Then
        assert len(similar_items) == 2
        assert "item2" in [item_id for item_id, _ in similar_items]
        # item2应该比item3更相似于item1
        item2_score = next(score for item_id, score in similar_items if item_id == "item2")
        item3_score = next(score for item_id, score in similar_items if item_id == "item3")
        assert item2_score > item3_score

    def test_find_similar_content_nonexistent_item(self):
        """测试查找相似内容 - 不存在的物品"""
        # Given & When
        similar_items = self.content_engine.find_similar_content("nonexistent", k=5)

        # Then
        assert len(similar_items) == 0

    def test_generate_recommendations(self):
        """测试生成推荐"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)
        self.content_engine.update_user_preference(user_id, "technology", 0.8)

        # 添加内容特征
        self.content_engine.add_content_feature("item1", "technology", {"ai": 0.8, "python": 0.7})
        self.content_engine.add_content_feature("item2", "technology", {"ai": 0.9, "python": 0.6})
        self.content_engine.add_content_feature("item3", "sports", {"football": 0.8})

        # When
        recommendations = self.content_engine.generate_recommendations(user_id, k=5)

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert recommendations.method == "content_based"
        assert len(recommendations.items) <= 5
        assert all(item.score > 0 for item in recommendations.items)

    def test_generate_recommendations_nonexistent_user(self):
        """测试生成推荐 - 不存在的用户"""
        # Given & When & Then
        with pytest.raises(ValueError, match="用户 nonexistent 不存在"):
            self.content_engine.generate_recommendations("nonexistent", k=5)

    def test_generate_recommendations_empty_profile(self):
        """测试生成推荐 - 空用户画像"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)  # 创建空画像

        # When
        recommendations = self.content_engine.generate_recommendations(user_id, k=5)

        # Then - 应该返回空推荐或基于内容特征的推荐
        assert isinstance(recommendations, RecommendationResult)
        assert recommendations.method == "content_based"

    def test_generate_recommendations_no_content(self):
        """测试生成推荐 - 无内容特征"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)
        self.content_engine.update_user_preference(user_id, "technology", 0.8)

        # When
        recommendations = self.content_engine.generate_recommendations(user_id, k=5)

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) == 0

    def test_recommendation_result_sorting(self):
        """测试推荐结果按分数排序"""
        # Given
        from recommendation_system.models import RecommendationItem
        items = [
            RecommendationItem("item3", 0.8, "High similarity"),
            RecommendationItem("item1", 0.9, "Very high similarity"),
            RecommendationItem("item2", 0.7, "Medium similarity")
        ]
        recommendations = RecommendationResult(
            user_id="user1",
            method="content_based",
            items=items
        )

        # When
        sorted_items = recommendations.get_sorted_items()

        # Then
        assert sorted_items[0].item_id == "item1"  # 最高分
        assert sorted_items[1].item_id == "item3"
        assert sorted_items[2].item_id == "item2"  # 最低分

    def test_get_popular_content(self):
        """测试获取热门内容"""
        # Given
        self.content_engine.add_content_feature("item1", "tech", {"ai": 0.8})
        self.content_engine.add_content_feature("item2", "tech", {"python": 0.7})
        self.content_engine.add_content_feature("item3", "sports", {"football": 0.9})

        # When
        popular_content = self.content_engine.get_popular_content(k=2)

        # Then
        assert len(popular_content) <= 2
        assert all(item_id in self.content_engine.content_features for item_id, _ in popular_content)

    def test_get_popular_content_empty(self):
        """测试获取热门内容 - 无内容"""
        # Given & When
        popular_content = self.content_engine.get_popular_content(k=5)

        # Then
        assert len(popular_content) == 0

    def test_clear_cache(self):
        """测试清空缓存"""
        # Given
        self.content_engine.add_content_feature("item1", "tech", {"ai": 0.8})
        self.content_engine.find_similar_content("item1")  # 这会填充缓存

        # When
        self.content_engine.clear_cache()

        # Then
        assert len(self.content_engine.similarity_cache) == 0

    def test_get_engine_stats(self):
        """测试获取引擎统计信息"""
        # Given
        self.content_engine.create_user_profile("user1")
        self.content_engine.add_content_feature("item1", "tech", {"ai": 0.8})

        # When
        stats = self.content_engine.get_engine_stats()

        # Then
        assert stats["num_users"] == 1
        assert stats["num_content_items"] == 1
        assert stats["cache_size"] == 0

    def test_update_feature_weights(self):
        """测试更新特征权重"""
        # Given
        new_weights = {"category": 0.4, "features": 0.5, "tags": 0.1}

        # When
        self.content_engine.update_feature_weights(new_weights)

        # Then
        assert self.content_engine.feature_weights == new_weights

    def test_update_feature_weights_invalid_weights(self):
        """测试更新特征权重 - 无效权重"""
        # Given
        invalid_weights = {"category": 0.5, "features": 0.6}  # 总和 > 1.0

        # When & Then
        with pytest.raises(ValueError, match="特征权重总和必须等于1.0"):
            self.content_engine.update_feature_weights(invalid_weights)

    def test_generate_recommendations_with_exclusions(self):
        """测试生成推荐 - 排除已交互物品"""
        # Given
        user_id = "user_1"
        self.content_engine.create_user_profile(user_id)
        self.content_engine.update_user_preference(user_id, "technology", 0.8)

        # 添加内容特征
        self.content_engine.add_content_feature("item1", "tech", {"ai": 0.8})
        self.content_engine.add_content_feature("item2", "tech", {"ai": 0.9})
        self.content_engine.add_content_feature("item3", "tech", {"python": 0.7})

        # 模拟用户已交互的物品
        excluded_items = {"item1"}

        # When
        recommendations = self.content_engine.generate_recommendations(
            user_id, k=5, exclude_items=excluded_items
        )

        # Then
        recommended_item_ids = {item.item_id for item in recommendations.items}
        assert "item1" not in recommended_item_ids  # 不应该推荐已交互的物品

    def test_performance_with_large_dataset(self):
        """测试大数据集性能"""
        # Given - 生成大数据集
        import time
        num_items = 1000
        num_features = 50

        # 创建用户画像
        self.content_engine.create_user_profile("user1")
        self.content_engine.update_user_preference("user1", "category", 0.8)

        for i in range(num_items):
            features = {f"feature_{j}": np.random.random() for j in range(num_features)}
            self.content_engine.add_content_feature(f"item_{i}", "category", features)

        # When
        start_time = time.time()
        recommendations = self.content_engine.generate_recommendations("user1", k=10)
        end_time = time.time()

        # Then
        processing_time = end_time - start_time
        assert processing_time < 5.0  # 应该在5秒内完成
        assert isinstance(recommendations, RecommendationResult)

    def test_content_feature_validation(self):
        """测试内容特征验证"""
        # Given
        item_id = "item1"
        category = "tech"
        features = {"ai": -0.5}  # 负特征值

        # When & Then
        with pytest.raises(ValueError, match="特征值不能为负数"):
            self.content_engine.add_content_feature(item_id, category, features)

    def test_user_profile_automatic_creation(self):
        """测试用户画像自动创建"""
        # Given
        user_id = "user1"

        # When - 尝试更新不存在用户的偏好
        self.content_engine.update_user_preference(user_id, "tech", 0.8)

        # Then - 应该自动创建用户画像
        assert user_id in self.content_engine.user_profiles
        profile = self.content_engine.user_profiles[user_id]
        assert profile.preferences["tech"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])