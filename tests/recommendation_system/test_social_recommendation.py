"""
社交推荐引擎的测试用例

使用TDD方法，先写测试用例，再实现功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from recommendation_system.social_recommendation import SocialRecommendationEngine
from recommendation_system.models import (
    UserProfile, RecommendationResult, RecommendationItem
)


class TestSocialRecommendationEngine:
    """社交推荐引擎测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.social_engine = SocialRecommendationEngine()

    def test_initialization(self):
        """测试社交推荐引擎初始化"""
        # Given & When
        engine = SocialRecommendationEngine()

        # Then
        assert engine is not None
        assert hasattr(engine, 'social_network')
        assert hasattr(engine, 'user_influence')
        assert hasattr(engine, 'trust_scores')
        assert hasattr(engine, 'recommendation_cache')

    def test_add_social_connection(self):
        """测试添加社交连接"""
        # Given
        user_a = "user1"
        user_b = "user2"
        strength = 0.8

        # When
        self.social_engine.add_social_connection(user_a, user_b, strength)

        # Then
        assert user_b in self.social_engine.social_network[user_a]
        assert self.social_engine.social_network[user_a][user_b] == strength
        # 社交关系应该是双向的
        assert user_a in self.social_engine.social_network[user_b]
        assert self.social_engine.social_network[user_b][user_a] == strength

    def test_add_social_connection_invalid_strength(self):
        """测试添加社交连接 - 无效强度"""
        # Given
        user_a = "user1"
        user_b = "user2"

        # When & Then
        with pytest.raises(ValueError, match="连接强度必须在0-1之间"):
            self.social_engine.add_social_connection(user_a, user_b, 1.5)

        with pytest.raises(ValueError, match="连接强度必须在0-1之间"):
            self.social_engine.add_social_connection(user_a, user_b, -0.1)

    def test_add_social_connection_self(self):
        """测试添加社交连接 - 自己连接自己"""
        # Given & When & Then
        with pytest.raises(ValueError, match="用户不能与自己建立连接"):
            self.social_engine.add_social_connection("user1", "user1", 0.5)

    def test_remove_social_connection(self):
        """测试移除社交连接"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)

        # When
        self.social_engine.remove_social_connection("user1", "user2")

        # Then
        assert "user2" not in self.social_engine.social_network["user1"]
        assert "user1" not in self.social_engine.social_network["user2"]

    def test_remove_nonexistent_connection(self):
        """测试移除不存在的社交连接"""
        # Given & When & Then - 应该不抛出异常
        self.social_engine.remove_social_connection("user1", "user2")  # 应该静默处理

    def test_update_user_influence(self):
        """测试更新用户影响力"""
        # Given
        user_id = "user1"
        influence_score = 0.85

        # When
        self.social_engine.update_user_influence(user_id, influence_score)

        # Then
        assert self.social_engine.user_influence[user_id] == influence_score

    def test_update_user_influence_invalid_score(self):
        """测试更新用户影响力 - 无效分数"""
        # Given & When & Then
        with pytest.raises(ValueError, match="影响力分数必须在0-1之间"):
            self.social_engine.update_user_influence("user1", 1.5)

    def test_calculate_social_influence(self):
        """测试计算社交影响力"""
        # Given
        # 构建社交网络：user1 -> user2 (0.8), user2 -> user3 (0.7), user1 -> user4 (0.6)
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.add_social_connection("user2", "user3", 0.7)
        self.social_engine.add_social_connection("user1", "user4", 0.6)

        # 设置用户影响力
        self.social_engine.update_user_influence("user1", 0.9)
        self.social_engine.update_user_influence("user2", 0.7)
        self.social_engine.update_user_influence("user3", 0.5)
        self.social_engine.update_user_influence("user4", 0.3)

        # When
        influence = self.social_engine._calculate_social_influence("user1", "user3")

        # Then
        assert 0.0 < influence < 1.0
        # user1通过user2对user3的影响力应该 > 直接连接的弱影响力

    def test_calculate_social_influence_no_path(self):
        """测试计算社交影响力 - 无路径"""
        # Given
        self.social_engine.update_user_influence("user1", 0.9)
        self.social_engine.update_user_influence("user2", 0.7)

        # When
        influence = self.social_engine._calculate_social_influence("user1", "user2")

        # Then
        assert influence == 0.0

    def test_get_friends_recommendations(self):
        """测试获取朋友推荐"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.add_social_connection("user1", "user3", 0.6)

        # 模拟朋友行为数据
        friends_activities = {
            "user2": {"item1": 5.0, "item2": 4.0},
            "user3": {"item1": 3.0, "item3": 5.0}
        }

        # When
        recommendations = self.social_engine._get_friends_recommendations(
            "user1", friends_activities, k=5
        )

        # Then
        assert len(recommendations) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in recommendations)
        assert all(isinstance(item_id, str) and isinstance(score, float)
                  for item_id, score in recommendations)

    def test_get_friends_recommendations_no_friends(self):
        """测试获取朋友推荐 - 无朋友"""
        # Given
        friends_activities = {}

        # When
        recommendations = self.social_engine._get_friends_recommendations(
            "user1", friends_activities, k=5
        )

        # Then
        assert len(recommendations) == 0

    def test_generate_recommendations(self):
        """测试生成推荐"""
        # Given
        user_id = "user1"

        # 构建社交网络
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.add_social_connection("user1", "user3", 0.6)

        # 设置用户影响力
        self.social_engine.update_user_influence("user2", 0.9)
        self.social_engine.update_user_influence("user3", 0.7)

        # 模拟用户行为数据
        user_activities = {
            "user2": {"item1": 5.0, "item2": 4.0},
            "user3": {"item1": 3.0, "item3": 5.0},
            "user1": {"item1": 2.0}  # 用户已交互的物品
        }

        # When
        recommendations = self.social_engine.generate_recommendations(
            user_id, user_activities, k=5
        )

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert recommendations.method == "social_based"
        assert len(recommendations.items) <= 5
        assert all(item.score > 0 for item in recommendations.items)

    def test_generate_recommendations_nonexistent_user(self):
        """测试生成推荐 - 不存在的用户"""
        # Given
        user_activities = {}

        # When
        recommendations = self.social_engine.generate_recommendations(
            "nonexistent", user_activities, k=5
        )

        # Then - 应该返回空推荐而不是抛出异常
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) == 0

    def test_generate_recommendations_no_activities(self):
        """测试生成推荐 - 无活动数据"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)

        # When
        recommendations = self.social_engine.generate_recommendations(
            "user1", {}, k=5
        )

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) == 0

    def test_calculate_trust_score(self):
        """测试计算信任分数"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.add_social_connection("user2", "user3", 0.7)

        # When
        trust_score = self.social_engine._calculate_trust_score("user1", "user3")

        # Then
        assert 0.0 <= trust_score <= 1.0

    def test_calculate_trust_score_direct_connection(self):
        """测试计算信任分数 - 直接连接"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.9)

        # When
        trust_score = self.social_engine._calculate_trust_score("user1", "user2")

        # Then
        assert trust_score == 0.9

    def test_calculate_trust_score_no_connection(self):
        """测试计算信任分数 - 无连接"""
        # Given & When
        trust_score = self.social_engine._calculate_trust_score("user1", "user2")

        # Then
        assert trust_score == 0.0

    def test_get_influential_users(self):
        """测试获取有影响力的用户"""
        # Given
        self.social_engine.update_user_influence("user1", 0.9)
        self.social_engine.update_user_influence("user2", 0.7)
        self.social_engine.update_user_influence("user3", 0.8)

        # When
        influential_users = self.social_engine.get_influential_users(k=2)

        # Then
        assert len(influential_users) == 2
        assert influential_users[0][0] == "user1"  # 最高影响力
        assert influential_users[1][0] == "user3"  # 第二高影响力
        # 检查按影响力降序排列
        assert influential_users[0][1] >= influential_users[1][1]

    def test_get_influential_users_empty(self):
        """测试获取有影响力的用户 - 空"""
        # Given & When
        influential_users = self.social_engine.get_influential_users(k=5)

        # Then
        assert len(influential_users) == 0

    def test_get_user_social_network(self):
        """测试获取用户社交网络"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.add_social_connection("user1", "user3", 0.6)
        self.social_engine.add_social_connection("user2", "user4", 0.7)

        # When
        network = self.social_engine.get_user_social_network("user1")

        # Then
        assert "user2" in network
        assert "user3" in network
        assert "user4" not in network  # 不是直接朋友
        assert network["user2"] == 0.8
        assert network["user3"] == 0.6

    def test_get_user_social_network_nonexistent(self):
        """测试获取用户社交网络 - 不存在的用户"""
        # Given & When
        network = self.social_engine.get_user_social_network("nonexistent")

        # Then
        assert len(network) == 0

    def test_clear_cache(self):
        """测试清空缓存"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        # 生成推荐以填充缓存
        user_activities = {"user2": {"item1": 5.0}}
        self.social_engine.generate_recommendations("user1", user_activities)

        # When
        self.social_engine.clear_cache()

        # Then
        assert len(self.social_engine.recommendation_cache) == 0

    def test_get_engine_stats(self):
        """测试获取引擎统计信息"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.update_user_influence("user1", 0.9)

        # When
        stats = self.social_engine.get_engine_stats()

        # Then
        assert stats["num_users"] == 2
        assert stats["num_connections"] == 1
        assert stats["cache_size"] >= 0

    def test_update_social_connection_strength(self):
        """测试更新社交连接强度"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.5)

        # When
        self.social_engine.add_social_connection("user1", "user2", 0.9)  # 更新

        # Then
        assert self.social_engine.social_network["user1"]["user2"] == 0.9
        assert self.social_engine.social_network["user2"]["user1"] == 0.9

    def test_get_social_paths(self):
        """测试获取社交路径"""
        # Given
        # user1 -> user2 -> user3 -> user4
        self.social_engine.add_social_connection("user1", "user2", 0.8)
        self.social_engine.add_social_connection("user2", "user3", 0.7)
        self.social_engine.add_social_connection("user3", "user4", 0.6)

        # When
        paths = self.social_engine._get_social_paths("user1", "user4", max_depth=3)

        # Then
        assert len(paths) > 0
        # 应该包含路径 user1 -> user2 -> user3 -> user4
        path_exists = any(
            len(path) == 3 and
            path[0] == "user2" and
            path[1] == "user3" and
            path[2] == "user4"
            for path in paths
        )
        assert path_exists

    def test_performance_with_large_network(self):
        """测试大数据集性能"""
        # Given - 生成大型社交网络
        import time
        num_users = 100
        connections_per_user = 10

        # 创建用户和连接
        for i in range(num_users):
            for j in range(connections_per_user):
                friend_id = f"user_{(i + j + 1) % num_users}"
                strength = np.random.random()
                self.social_engine.add_social_connection(f"user_{i}", friend_id, strength)

        # 设置用户影响力
        for i in range(num_users):
            influence = np.random.random()
            self.social_engine.update_user_influence(f"user_{i}", influence)

        # 准备用户活动数据
        user_activities = {}
        for i in range(10):  # 只测试10个用户的活动
            user_id = f"user_{i}"
            user_activities[user_id] = {
                f"item_{j}": np.random.uniform(1, 5)
                for j in range(20)
            }

        # When
        start_time = time.time()
        recommendations = self.social_engine.generate_recommendations(
            "user_0", user_activities, k=10
        )
        end_time = time.time()

        # Then
        processing_time = end_time - start_time
        assert processing_time < 10.0  # 应该在10秒内完成
        assert isinstance(recommendations, RecommendationResult)

    def test_social_network_validation(self):
        """测试社交网络数据验证"""
        # Given
        # 尝试添加无效的用户ID
        invalid_cases = [
            ("", "user2", 0.5),  # 空用户ID
            ("user1", "", 0.5),   # 空朋友ID
        ]

        # When & Then
        for user_a, user_b, strength in invalid_cases:
            with pytest.raises(ValueError, match="用户ID不能为空"):
                self.social_engine.add_social_connection(user_a, user_b, strength)

    def test_recommendation_excluding_user_items(self):
        """测试推荐排除用户已交互物品"""
        # Given
        self.social_engine.add_social_connection("user1", "user2", 0.8)

        user_activities = {
            "user1": {"item1": 5.0, "item2": 3.0},  # 用户已交互物品
            "user2": {"item1": 4.0, "item2": 2.0, "item3": 5.0, "item4": 4.0}  # 朋友交互物品
        }

        # When
        recommendations = self.social_engine.generate_recommendations(
            "user1", user_activities, k=5
        )

        # Then
        recommended_items = {item.item_id for item in recommendations.items}
        # 不应该推荐用户已交互的物品
        assert "item1" not in recommended_items
        assert "item2" not in recommended_items
        # 但可以推荐朋友交互但用户未交互的物品
        assert any(item in recommended_items for item in ["item3", "item4"])


if __name__ == "__main__":
    pytest.main([__file__])