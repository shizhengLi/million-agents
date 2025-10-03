"""
混合推荐引擎的测试用例

使用TDD方法，先写测试用例，再实现功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from recommendation_system.hybrid_recommendation import HybridRecommendationEngine
from recommendation_system.models import (
    RecommendationResult, RecommendationItem, UserProfile
)


class TestHybridRecommendationEngine:
    """混合推荐引擎测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.hybrid_engine = HybridRecommendationEngine()

    def test_initialization(self):
        """测试混合推荐引擎初始化"""
        # Given & When
        engine = HybridRecommendationEngine()

        # Then
        assert engine is not None
        assert hasattr(engine, 'collaborative_engine')
        assert hasattr(engine, 'content_engine')
        assert hasattr(engine, 'social_engine')
        assert hasattr(engine, 'recommendation_weights')
        assert hasattr(engine, 'recommendation_cache')

    def test_initialization_with_engines(self):
        """测试混合推荐引擎初始化 - 指定引擎"""
        # Given
        cf_engine = Mock()
        content_engine = Mock()
        social_engine = Mock()

        # When
        engine = HybridRecommendationEngine(cf_engine, content_engine, social_engine)

        # Then
        assert engine.collaborative_engine == cf_engine
        assert engine.content_engine == content_engine
        assert engine.social_engine == social_engine

    def test_update_recommendation_weights(self):
        """测试更新推荐权重"""
        # Given
        new_weights = {
            "collaborative": 0.4,
            "content": 0.3,
            "social": 0.3
        }

        # When
        self.hybrid_engine.update_recommendation_weights(new_weights)

        # Then
        assert self.hybrid_engine.recommendation_weights == new_weights

    def test_update_recommendation_weights_invalid_sum(self):
        """测试更新推荐权重 - 权重总和不等于1"""
        # Given
        invalid_weights = {
            "collaborative": 0.5,
            "content": 0.3,
            "social": 0.3  # 总和=1.1
        }

        # When & Then
        with pytest.raises(ValueError, match="推荐权重总和必须等于1.0"):
            self.hybrid_engine.update_recommendation_weights(invalid_weights)

    def test_update_recommendation_weights_invalid_keys(self):
        """测试更新推荐权重 - 无效键"""
        # Given
        invalid_weights = {
            "collaborative": 0.4,
            "content": 0.3,
            "invalid": 0.3  # 无效键
        }

        # When & Then
        with pytest.raises(ValueError, match="必须包含所有必需的推荐引擎权重"):
            self.hybrid_engine.update_recommendation_weights(invalid_weights)

    def test_calculate_hybrid_score(self):
        """测试计算混合分数"""
        # Given
        cf_score = 0.8
        content_score = 0.6
        social_score = 0.7

        # When
        hybrid_score = self.hybrid_engine._calculate_hybrid_score(
            cf_score, content_score, social_score
        )

        # Then
        expected_score = (
            0.5 * cf_score +  # 默认权重
            0.3 * content_score +
            0.2 * social_score
        )
        assert abs(hybrid_score - expected_score) < 1e-6
        assert 0.0 <= hybrid_score <= 1.0

    def test_calculate_hybrid_score_with_none(self):
        """测试计算混合分数 - 包含None值"""
        # Given
        cf_score = 0.8
        content_score = None  # 无内容推荐
        social_score = 0.7

        # When
        hybrid_score = self.hybrid_engine._calculate_hybrid_score(
            cf_score, content_score, social_score
        )

        # Then - 应该忽略None值，重新分配权重
        total_weight = 0.5 + 0.2  # cf + social
        expected_score = (0.5 * cf_score + 0.2 * social_score) / total_weight
        assert abs(hybrid_score - expected_score) < 1e-6

    def test_calculate_hybrid_score_all_none(self):
        """测试计算混合分数 - 全部为None"""
        # Given & When
        hybrid_score = self.hybrid_engine._calculate_hybrid_score(None, None, None)

        # Then
        assert hybrid_score == 0.0

    def test_generate_recommendations_basic(self):
        """测试生成推荐 - 基本功能"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0), ("user2", "item1", 4.0)]
        user_activities = {"user1": {"item1": 5.0}, "user2": {"item1": 4.0}}

        # Mock各个引擎的返回结果
        cf_result = RecommendationResult(user_id, "collaborative", [
            RecommendationItem("item2", 0.8), RecommendationItem("item3", 0.6)
        ])
        content_result = RecommendationResult(user_id, "content", [
            RecommendationItem("item3", 0.7), RecommendationItem("item4", 0.5)
        ])
        social_result = RecommendationResult(user_id, "social", [
            RecommendationItem("item4", 0.6), RecommendationItem("item5", 0.4)
        ])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert recommendations.method == "hybrid"
        assert len(recommendations.items) <= 5
        assert all(item.score > 0 for item in recommendations.items)
        # 验证推荐包含不同引擎的结果
        item_ids = {item.item_id for item in recommendations.items}
        assert "item2" in item_ids or "item3" in item_ids or "item4" in item_ids

    def test_generate_recommendations_empty_interactions(self):
        """测试生成推荐 - 空交互数据"""
        # Given
        user_id = "user1"
        interactions = []
        user_activities = {}

        # Mock引擎返回空结果
        cf_result = RecommendationResult(user_id, "collaborative", [])
        content_result = RecommendationResult(user_id, "content", [])
        social_result = RecommendationResult(user_id, "social", [])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) == 0

    def test_generate_recommendations_with_engine_failures(self):
        """测试生成推荐 - 引擎失败处理"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # 模拟协同过滤引擎失败
        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(
            side_effect=Exception("CF Engine Error")
        )

        # 其他引擎正常
        content_result = RecommendationResult(user_id, "content", [
            RecommendationItem("item2", 0.8)
        ])
        social_result = RecommendationResult(user_id, "social", [
            RecommendationItem("item3", 0.7)
        ])

        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # Then - 应该使用其他引擎的结果
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) > 0
        item_ids = {item.item_id for item in recommendations.items}
        assert "item2" in item_ids or "item3" in item_ids

    def test_generate_recommendations_weighted_combination(self):
        """测试生成推荐 - 权重组合"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # 设置权重
        weights = {"collaborative": 0.6, "content": 0.3, "social": 0.1}
        self.hybrid_engine.update_recommendation_weights(weights)

        # Mock引擎结果
        cf_result = RecommendationResult(user_id, "collaborative", [
            RecommendationItem("item2", 0.9), RecommendationItem("item3", 0.8)
        ])
        content_result = RecommendationResult(user_id, "content", [
            RecommendationItem("item3", 0.7), RecommendationItem("item4", 0.6)
        ])
        social_result = RecommendationResult(user_id, "social", [
            RecommendationItem("item4", 0.5), RecommendationItem("item5", 0.4)
        ])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # Then - 验证权重影响
        assert isinstance(recommendations, RecommendationResult)
        # item2应该排名靠前，因为协同过滤权重最高
        item_ids = [item.item_id for item in recommendations.items]
        item2_index = item_ids.index("item2") if "item2" in item_ids else -1
        item4_index = item_ids.index("item4") if "item4" in item_ids else -1

        if item2_index != -1 and item4_index != -1:
            assert item2_index < item4_index  # item2应该排名更靠前

    def test_get_recommendation_explanation(self):
        """测试获取推荐解释"""
        # Given
        user_id = "user1"
        item_id = "item2"

        # Mock引擎返回结果
        cf_result = RecommendationResult(user_id, "collaborative", [
            RecommendationItem("item2", 0.8)
        ])
        content_result = RecommendationResult(user_id, "content", [
            RecommendationItem("item2", 0.6)
        ])
        social_result = RecommendationResult(user_id, "social", [])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        explanation = self.hybrid_engine.get_recommendation_explanation(
            user_id, item_id, [], {}
        )

        # Then
        assert isinstance(explanation, dict)
        assert "engines" in explanation
        assert "final_score" in explanation
        assert "weighting" in explanation
        assert len(explanation["engines"]) >= 2  # 至少有两个引擎有结果

    def test_get_recommendation_explanation_item_not_recommended(self):
        """测试获取推荐解释 - 物品未被推荐"""
        # Given
        user_id = "user1"
        item_id = "nonexistent"

        # Mock引擎返回空结果
        cf_result = RecommendationResult(user_id, "collaborative", [])
        content_result = RecommendationResult(user_id, "content", [])
        social_result = RecommendationResult(user_id, "social", [])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        explanation = self.hybrid_engine.get_recommendation_explanation(
            user_id, item_id, [], {}
        )

        # Then
        assert explanation["final_score"] == 0.0
        assert len(explanation["engines"]) == 0

    def test_clear_cache(self):
        """测试清空缓存"""
        # Given
        self.hybrid_engine.recommendation_cache["test"] = "value"

        # When
        self.hybrid_engine.clear_cache()

        # Then
        assert len(self.hybrid_engine.recommendation_cache) == 0

    def test_get_engine_stats(self):
        """测试获取引擎统计信息"""
        # Given & When
        stats = self.hybrid_engine.get_engine_stats()

        # Then
        assert isinstance(stats, dict)
        assert "collaborative_engine" in stats
        assert "content_engine" in stats
        assert "social_engine" in stats
        assert "hybrid_weights" in stats
        assert "cache_size" in stats

    def test_adaptive_weight_adjustment(self):
        """测试自适应权重调整"""
        # Given
        user_id = "user1"
        performance_metrics = {
            "collaborative": {"precision": 0.8, "recall": 0.7},
            "content": {"precision": 0.6, "recall": 0.5},
            "social": {"precision": 0.4, "recall": 0.3}
        }

        # When
        self.hybrid_engine.adaptive_weight_adjustment(user_id, performance_metrics)

        # Then - 权重应该根据性能调整
        weights = self.hybrid_engine.recommendation_weights
        # 协同过滤应该有更高的权重（性能更好）
        assert weights["collaborative"] > weights["content"]
        assert weights["content"] > weights["social"]
        # 权重总和应该还是1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_fallback_recommendation(self):
        """测试回退推荐机制"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # 所有引擎都失败
        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(
            side_effect=Exception("CF Error")
        )
        self.hybrid_engine.content_engine.generate_recommendations = Mock(
            side_effect=Exception("Content Error")
        )
        self.hybrid_engine.social_engine.generate_recommendations = Mock(
            side_effect=Exception("Social Error")
        )

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # Then - 应该有回退机制
        assert isinstance(recommendations, RecommendationResult)
        # 可能返回基于热门的推荐或空推荐

    def test_personalized_weight_learning(self):
        """测试个性化权重学习"""
        # Given
        user_id = "user1"
        historical_feedback = [
            {"item_id": "item1", "rating": 5.0, "source": "collaborative"},
            {"item_id": "item2", "rating": 3.0, "source": "content"},
            {"item_id": "item3", "rating": 4.0, "source": "social"}
        ]

        # When
        learned_weights = self.hybrid_engine.learn_personalized_weights(
            user_id, historical_feedback
        )

        # Then
        assert isinstance(learned_weights, dict)
        assert "collaborative" in learned_weights
        assert "content" in learned_weights
        assert "social" in learned_weights
        total_weight = sum(learned_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_diversity_enhancement(self):
        """测试多样性增强"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # Mock引擎返回相似的推荐
        cf_result = RecommendationResult(user_id, "collaborative", [
            RecommendationItem("tech_item1", 0.9),
            RecommendationItem("tech_item2", 0.8),
            RecommendationItem("tech_item3", 0.7)
        ])
        content_result = RecommendationResult(user_id, "content", [
            RecommendationItem("tech_item1", 0.8),
            RecommendationItem("sports_item1", 0.6),
            RecommendationItem("music_item1", 0.5)
        ])
        social_result = RecommendationResult(user_id, "social", [
            RecommendationItem("tech_item1", 0.7),
            RecommendationItem("sports_item2", 0.6)
        ])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5, enhance_diversity=True
        )

        # Then - 应该包含不同类别的物品
        item_ids = [item.item_id for item in recommendations.items]
        categories = [item_id.split("_")[0] for item_id in item_ids]
        unique_categories = set(categories)
        assert len(unique_categories) >= 2  # 至少有两个不同类别

    def test_performance_with_large_dataset(self):
        """测试大数据集性能"""
        # Given
        import time
        user_id = "user1"
        num_interactions = 1000
        interactions = [(f"user_{i%10}", f"item_{i}", np.random.uniform(1, 5))
                       for i in range(num_interactions)]

        user_activities = {}
        for i in range(10):
            user_id_i = f"user_{i}"
            user_activities[user_id_i] = {
                f"item_{j}": np.random.uniform(1, 5)
                for j in range(i*10, (i+1)*10)
            }

        # Mock引擎返回
        def mock_generate_recommendations(*args, **kwargs):
            return RecommendationResult(
                user_id, "mock",
                [RecommendationItem(f"item_{i}", np.random.random()) for i in range(20)]
            )

        self.hybrid_engine.collaborative_engine.generate_recommendations = mock_generate_recommendations
        self.hybrid_engine.content_engine.generate_recommendations = mock_generate_recommendations
        self.hybrid_engine.social_engine.generate_recommendations = mock_generate_recommendations

        # When
        start_time = time.time()
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=10
        )
        end_time = time.time()

        # Then
        processing_time = end_time - start_time
        assert processing_time < 5.0  # 应该在5秒内完成
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) <= 10

    def test_cold_start_handling(self):
        """测试冷启动处理"""
        # Given
        new_user_id = "new_user"
        interactions = []  # 新用户无历史数据
        user_activities = {}

        # Mock协同过滤引擎因冷启动返回空结果
        cf_result = RecommendationResult(new_user_id, "collaborative", [])
        # 内容和社交引擎可以基于其他信息提供推荐
        content_result = RecommendationResult(new_user_id, "content", [
            RecommendationItem("popular_item1", 0.8),
            RecommendationItem("popular_item2", 0.7)
        ])
        social_result = RecommendationResult(new_user_id, "social", [
            RecommendationItem("trending_item1", 0.6)
        ])

        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=content_result)
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=social_result)

        # When
        recommendations = self.hybrid_engine.generate_recommendations(
            new_user_id, interactions, user_activities, k=5
        )

        # Then - 应该提供基于内容和社交的推荐
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) > 0
        item_ids = {item.item_id for item in recommendations.items}
        assert "popular_item1" in item_ids or "trending_item1" in item_ids

    def test_real_time_recommendation_updates(self):
        """测试实时推荐更新"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # 初始推荐
        cf_result = RecommendationResult(user_id, "collaborative", [
            RecommendationItem("item2", 0.8)
        ])
        self.hybrid_engine.collaborative_engine.generate_recommendations = Mock(return_value=cf_result)
        self.hybrid_engine.content_engine.generate_recommendations = Mock(return_value=RecommendationResult(user_id, "content", []))
        self.hybrid_engine.social_engine.generate_recommendations = Mock(return_value=RecommendationResult(user_id, "social", []))

        # When - 用户交互新物品后
        initial_recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # 更新用户活动
        user_activities["user1"]["item2"] = 4.0

        # 清空缓存强制重新计算
        self.hybrid_engine.clear_cache()

        updated_recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5
        )

        # Then - 推荐应该更新
        assert isinstance(initial_recommendations, RecommendationResult)
        assert isinstance(updated_recommendations, RecommendationResult)

    def test_multi_criteria_recommendation(self):
        """测试多标准推荐"""
        # Given
        user_id = "user1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # When - 带有多标准约束
        recommendations = self.hybrid_engine.generate_recommendations(
            user_id, interactions, user_activities, k=5,
            criteria={"category": "technology", "min_rating": 4.0, "exclude_seen": True}
        )

        # Then - 推荐应该满足标准约束
        assert isinstance(recommendations, RecommendationResult)
        # 实际实现中会过滤不符合标准的推荐

    def test_learn_personalized_weights_with_insufficient_data(self):
        """测试学习个性化权重 - 数据不足的情况"""
        # Given
        user_id = "user1"
        historical_feedback = []  # 空反馈历史

        # When
        learned_weights = self.hybrid_engine.learn_personalized_weights(
            user_id, historical_feedback
        )

        # Then
        assert isinstance(learned_weights, dict)
        assert all(engine in learned_weights for engine in ["collaborative", "content", "social"])
        assert abs(sum(learned_weights.values()) - 1.0) < 1e-6

    def test_adaptive_weight_adjustment_with_zero_metrics(self):
        """测试自适应权重调整 - 零性能指标"""
        # Given
        user_id = "user1"
        performance_metrics = {
            "collaborative": {"precision": 0, "recall": 0},
            "content": {"precision": 0, "recall": 0},
            "social": {"precision": 0, "recall": 0}
        }

        # When
        self.hybrid_engine.adaptive_weight_adjustment(user_id, performance_metrics)

        # Then
        assert user_id in self.hybrid_engine.personalized_weights

    def test_apply_criteria_with_category_filtering(self):
        """测试应用推荐标准 - 类别过滤"""
        # Given
        recommendations = [
            RecommendationItem("action_item1", 0.9),
            RecommendationItem("comedy_item1", 0.8),
            RecommendationItem("drama_item1", 0.7)
        ]
        criteria = {"category": "action"}
        user_activities = {}

        # When
        filtered = self.hybrid_engine._apply_criteria(
            recommendations, criteria, user_activities, "user1"
        )

        # Then
        assert len(filtered) == 1
        assert filtered[0].item_id == "action_item1"

    def test_apply_criteria_with_exclude_seen(self):
        """测试应用推荐标准 - 排除已见过的物品"""
        # Given
        recommendations = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8),
            RecommendationItem("item3", 0.7)
        ]
        criteria = {"exclude_seen": True}
        user_activities = {"user1": {"item1": 5.0, "item3": 3.0}}

        # When
        filtered = self.hybrid_engine._apply_criteria(
            recommendations, criteria, user_activities, "user1"
        )

        # Then
        assert len(filtered) == 1
        assert filtered[0].item_id == "item2"

    def test_apply_criteria_with_min_rating(self):
        """测试应用推荐标准 - 最小评分过滤"""
        # Given
        recommendations = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8),
            RecommendationItem("item3", 0.7)
        ]
        # 为推荐项添加rating属性
        recommendations[0].rating = 4.5
        recommendations[1].rating = 3.5
        recommendations[2].rating = 2.5

        criteria = {"min_rating": 4.0}
        user_activities = {}

        # When
        filtered = self.hybrid_engine._apply_criteria(
            recommendations, criteria, user_activities, "user1"
        )

        # Then
        assert len(filtered) == 1
        assert filtered[0].item_id == "item1"

    def test_get_recommendation_explanation_with_all_engines(self):
        """测试获取推荐解释 - 所有引擎都有贡献"""
        # Given
        user_id = "user1"
        item_id = "item1"
        interactions = [("user1", "item1", 5.0)]
        user_activities = {"user1": {"item1": 5.0}}

        # When
        explanation = self.hybrid_engine.get_recommendation_explanation(
            user_id, item_id, interactions, user_activities
        )

        # Then
        assert explanation["item_id"] == item_id
        assert "engines" in explanation
        assert "final_score" in explanation
        assert "weighting" in explanation
        assert isinstance(explanation["final_score"], float)

    def test_get_recommendation_explanation_partial_engines(self):
        """测试获取推荐解释 - 部分引擎有贡献"""
        # Given
        user_id = "nonexistent_user"
        item_id = "item1"
        interactions = []
        user_activities = {}

        # When
        explanation = self.hybrid_engine.get_recommendation_explanation(
            user_id, item_id, interactions, user_activities
        )

        # Then
        assert explanation["item_id"] == item_id
        assert explanation["engines"] == {}
        assert explanation["final_score"] == 0.0

    def test_enhance_diversity_with_insufficient_items(self):
        """测试增强多样性 - 物品数量不足"""
        # Given
        recommendations = [
            RecommendationItem("item1", 0.9)
        ]
        k = 5

        # When
        diverse = self.hybrid_engine._enhance_diversity(recommendations, k)

        # Then
        assert len(diverse) == 1

    def test_enhance_diversity_no_categories(self):
        """测试增强多样性 - 无类别区分"""
        # Given
        recommendations = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8),
            RecommendationItem("item3", 0.7)
        ]
        k = 2

        # When
        diverse = self.hybrid_engine._enhance_diversity(recommendations, k)

        # Then
        assert len(diverse) == 2
        # 应该选择分数最高的两个
        assert diverse[0].item_id == "item1"
        assert diverse[1].item_id == "item2"

    def test_combine_recommendations_all_engines_failed(self):
        """测试合并推荐结果 - 所有引擎失败"""
        # Given
        engine_results = {
            "collaborative": None,
            "content": None,
            "social": None
        }
        user_id = "user1"
        k = 5

        # When
        combined = self.hybrid_engine._combine_recommendations(
            engine_results, user_id, k
        )

        # Then
        assert len(combined) == 0

    def test_combine_recommendations_partial_engines(self):
        """测试合并推荐结果 - 部分引擎成功"""
        # Given
        cf_result = RecommendationResult("user1", "collaborative", [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8)
        ])
        engine_results = {
            "collaborative": cf_result,
            "content": None,
            "social": None
        }
        user_id = "user1"
        k = 5

        # When
        combined = self.hybrid_engine._combine_recommendations(
            engine_results, user_id, k
        )

        # Then
        assert len(combined) == 2
        assert combined[0].item_id == "item1"
        assert combined[1].item_id == "item2"

    def test_calculate_hybrid_score_none_inputs(self):
        """测试计算混合分数 - 全部为None"""
        # Given
        cf_score = None
        content_score = None
        social_score = None

        # When
        hybrid_score = self.hybrid_engine._calculate_hybrid_score(
            cf_score, content_score, social_score
        )

        # Then
        assert hybrid_score == 0.0

    def test_learn_personalized_weights_unknown_source(self):
        """测试学习个性化权重 - 未知来源的反馈"""
        # Given
        user_id = "user1"
        historical_feedback = [
            {"rating": 5.0, "source": "unknown_engine"},
            {"rating": 3.0, "source": "another_unknown"}
        ]

        # When
        learned_weights = self.hybrid_engine.learn_personalized_weights(
            user_id, historical_feedback
        )

        # Then
        assert isinstance(learned_weights, dict)
        assert all(engine in learned_weights for engine in ["collaborative", "content", "social"])
        assert abs(sum(learned_weights.values()) - 1.0) < 1e-6

    def test_edge_case_zero_total_weight_in_hybrid_score(self):
        """测试边界情况 - 混合分数计算中总权重为零"""
        # Given
        # 创建一个特殊的权重配置来模拟总权重为零的情况
        original_weights = self.hybrid_engine.recommendation_weights.copy()
        self.hybrid_engine.recommendation_weights = {
            "collaborative": 0.0,
            "content": 0.0,
            "social": 0.0
        }

        # When
        hybrid_score = self.hybrid_engine._calculate_hybrid_score(
            0.5, 0.6, 0.7
        )

        # Then
        assert hybrid_score == 0.0

        # Cleanup
        self.hybrid_engine.recommendation_weights = original_weights

    def test_criteria_filtering_invalid_user_activities(self):
        """测试标准过滤 - 无效的用户活动数据"""
        # Given
        recommendations = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8)
        ]
        criteria = {"exclude_seen": True}
        user_activities = {"invalid_user": {"item1": 5.0}}  # 不包含当前用户

        # When
        filtered = self.hybrid_engine._apply_criteria(
            recommendations, criteria, user_activities, "user1"
        )

        # Then
        # 应该返回所有推荐，因为用户活动数据中没有当前用户
        assert len(filtered) == 2

    def test_personalized_weights_storage_and_retrieval(self):
        """测试个性化权重的存储和检索"""
        # Given
        user_id = "user1"
        custom_weights = {"collaborative": 0.6, "content": 0.3, "social": 0.1}

        # When
        self.hybrid_engine.personalized_weights[user_id] = custom_weights

        # Then
        assert user_id in self.hybrid_engine.personalized_weights
        assert self.hybrid_engine.personalized_weights[user_id] == custom_weights

    def test_engine_stats_with_personalized_weights(self):
        """测试引擎统计 - 包含个性化权重信息"""
        # Given
        self.hybrid_engine.personalized_weights["user1"] = {
            "collaborative": 0.6, "content": 0.3, "social": 0.1
        }

        # When
        stats = self.hybrid_engine.get_engine_stats()

        # Then
        assert stats["personalized_weights_count"] == 1
        assert "hybrid_weights" in stats
        assert "cache_size" in stats


if __name__ == "__main__":
    pytest.main([__file__])