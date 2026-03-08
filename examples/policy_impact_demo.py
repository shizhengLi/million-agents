#!/usr/bin/env python3
"""
虚拟社会政策影响分析Demo
模拟某项政策在虚拟社会中的接受度和传播效果
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.batch_manager import BatchAgentManager
from message_propagation.information_diffusion import InformationDiffusionModel
from message_propagation.social_network import MockSocialNetwork
import random


def simulate_policy_impact():
    """模拟政策在虚拟社会中的影响"""

    print("🏛️ 虚拟社会政策影响分析")
    print("=" * 50)

    # 1. 创建多元化虚拟社会
    print("👥 创建虚拟社会群体...")
    manager = BatchAgentManager(max_agents=5000, batch_size=200)

    # 创建不同背景的智能体
    social_groups = {
        "年轻人": {
            "count": 150,
            "interests": ["科技", "创新", "环保"],
            "personality": "curious",
        },
        "中年人": {
            "count": 200,
            "interests": ["经济", "家庭", "教育"],
            "personality": "casual",
        },
        "老年人": {
            "count": 100,
            "interests": ["健康", "传统", "安全"],
            "personality": "formal",
        },
        "专业人士": {
            "count": 50,
            "interests": ["专业", "发展", "政策"],
            "personality": "analytical",
        },
    }

    all_agents = []
    for group_name, config in social_groups.items():
        agents = manager.create_batch_agents(
            count=config["count"],
            name_prefix=group_name,
            personalities=[config["personality"]],
            interests_list=[config["interests"]],
        )
        all_agents.extend(agents)
        print(f"   创建了 {len(agents)} 个{group_name}")

    # 2. 构建社会网络
    print("🕸️ 构建社会网络关系...")
    # 同类群体内部连接更紧密
    for group_name, config in social_groups.items():
        group_agents = [a for a in all_agents if a.personality == config["personality"]]
        for agent in group_agents[: len(group_agents) // 2]:  # 只处理前一半避免重复
            # 同群体好友 (70%概率)
            for other in random.sample(group_agents, min(5, len(group_agents) - 1)):
                if other != agent and random.random() < 0.7:
                    agent.add_friend(other)

            # 跨群体好友 (30%概率)
            other_groups = [
                a for a in all_agents if a.personality != config["personality"]
            ]
            if other_groups and random.random() < 0.3:
                friend = random.choice(other_groups)
                agent.add_friend(friend)

    # 3. 模拟不同政策传播
    policies = [
        {
            "name": "环保税政策",
            "message": "政府将实施新的环保税收政策，旨在减少碳排放，促进绿色发展",
            "appeal_groups": ["年轻人", "专业人士"],
            "resistance_groups": ["中年人", "老年人"],
        },
        {
            "name": "数字化教育改革",
            "message": "全面推进教育数字化转型，提供在线学习资源和智能教学工具",
            "appeal_groups": ["年轻人", "专业人士", "中年人"],
            "resistance_groups": ["老年人"],
        },
    ]

    # 创建网络结构
    agents_data = {
        a.agent_id: {"id": a.agent_id, "name": a.name, "type": a.personality}
        for a in all_agents
    }
    connections = {
        a.agent_id: [(friend, 0.6) for friend in a.friends] for a in all_agents
    }
    social_network = MockSocialNetwork(list(agents_data.keys()), connections)

    # 4. 模拟政策传播
    results = {}

    for policy in policies:
        print(f"\n📢 模拟{policy['name']}传播...")

        # 根据政策调整采用概率
        base_prob = 0.3
        if policy["name"] == "环保税政策":
            base_prob = 0.25
        elif policy["name"] == "数字化教育改革":
            base_prob = 0.35

        diffusion_model = InformationDiffusionModel(
            social_network,
            adoption_probability=base_prob,
            abandon_probability=0.05,
            max_time_steps=15,
        )

        # 选择不同群体的种子
        seeds = []
        for group in policy["appeal_groups"]:
            group_agents = [
                a
                for a in all_agents
                if a.personality == social_groups[group]["personality"]
            ]
            seeds.extend(
                [
                    a.agent_id
                    for a in random.sample(group_agents, min(10, len(group_agents)))
                ]
            )

        # 设置初始采用者
        diffusion_model.set_initial_adopters(seeds)

        # 预测扩散
        diffusion_model.predict_diffusion()

        # 获取统计信息
        stats = diffusion_model.get_diffusion_statistics()

        # 分析结果
        acceptance_rate = stats["adoption_rate"] * 100
        print(f"   总接受度: {acceptance_rate:.1f}%")
        print(f"   传播步数: {stats['diffusion_steps']}")

        # 按群体分析
        group_analysis = {}
        for group_name, config in social_groups.items():
            group_agents = [
                a.agent_id for a in all_agents if a.personality == config["personality"]
            ]
            influenced_in_group = len(
                set(diffusion_model.adopted_agents) & set(group_agents)
            )
            group_rate = influenced_in_group / len(group_agents) * 100
            group_analysis[group_name] = group_rate
            print(f"   {group_name}接受度: {group_rate:.1f}%")

        results[policy["name"]] = {
            "adoption_rate": stats["adoption_rate"],
            "adopted_agents": diffusion_model.adopted_agents,
            "total_steps": stats["diffusion_steps"],
            "stats": stats,
        }

    # 5. 政策建议
    print("\n💡 政策实施建议:")

    for policy_name, result in results.items():
        acceptance_rate = result["adoption_rate"] * 100

        if acceptance_rate > 60:
            print(f"\n✅ {policy_name}: 高接受度政策")
            print("   建议: 可以直接实施，预期社会阻力较小")
        elif acceptance_rate > 40:
            print(f"\n⚠️ {policy_name}: 中等接受度政策")
            print("   建议: 需要加强宣传和解释，针对特定群体制定沟通策略")
        else:
            print(f"\n❌ {policy_name}: 低接受度政策")
            print("   建议: 需要重新设计政策内容，增加配套措施")

    return results


if __name__ == "__main__":
    simulate_policy_impact()
