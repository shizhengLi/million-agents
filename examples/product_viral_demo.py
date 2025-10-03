#!/usr/bin/env python3
"""
社交网络传播研究Demo
模拟一个新产品在社交网络中的传播过程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.batch_manager import BatchAgentManager
from message_propagation.viral_propagation import ViralPropagationModel
from message_propagation.social_network import MockSocialNetwork

def simulate_product_viral_marketing():
    """模拟产品病毒式营销传播"""

    print("🚀 产品病毒式营销传播模拟")
    print("=" * 50)

    # 1. 创建目标用户群体 (1000个智能体)
    print("👥 创建目标用户群体...")
    manager = BatchAgentManager(max_agents=2000, batch_size=100)

    # 创建不同类型的用户
    users = manager.create_batch_agents(
        count=1000,
        name_prefix="User",
        personalities=["analytical", "friendly", "casual"],
        interests_list=[["科技", "生活", "娱乐", "体育", "美食"]]
    )

    # 2. 建立社交网络关系
    print("🕸️ 构建社交网络...")
    friendships = manager.create_random_friendships(max_friends_per_agent=8)

    # 创建社交网络数据结构
    agents_data = {}
    connections = {}

    for user in users:
        agents_data[user.agent_id] = {
            'id': user.agent_id,
            'name': user.name,
            'type': user.personality,
            'status': 'active'
        }
        connections[user.agent_id] = [(friend, 0.8) for friend in user.friends]

    social_network = MockSocialNetwork(
        agents=list(agents_data.keys()),
        adjacency_list=connections
    )

    # 3. 设置传播参数
    print("📢 设置产品传播参数...")
    propagation_model = ViralPropagationModel(social_network)

    # 不同类型用户的传播概率
    infection_probs = {
        "analytical": 0.8,  # 分析型用户更容易接受和传播
        "friendly": 0.3,     # 友好型普通用户
        "casual": 0.1      # 随意型用户不容易接受
    }

    # 4. 选择种子用户 (早期采用者)
    print("🌱 选择种子用户...")
    seed_users = [user for user in users if user.personality == "analytical"][:10]
    seed_ids = [user.agent_id for user in seed_users]

    print(f"   选择了 {len(seed_ids)} 个影响者作为种子用户")

    # 5. 模拟传播过程
    print("📈 开始传播模拟...")

    class CustomPropagationModel(ViralPropagationModel):
        def __init__(self, network, infection_probs):
            super().__init__(network)
            self.infection_probs = infection_probs

        def get_infection_probability(self, agent_id, step):
            # 根据智能体类型返回不同的感染概率
            agent = self.network.get_agent(agent_id)
            agent_type = agent.get('type', 'regular') if agent else 'regular'
            return self.infection_probs.get(agent_type, 0.3)

    custom_model = CustomPropagationModel(social_network, infection_probs)

    # 执行传播
    custom_model.set_initial_infected(seed_ids)
    custom_model.max_iterations = 20
    history = custom_model.propagate_full_simulation()

    # 6. 分析结果
    print("📊 传播结果分析...")
    total_influenced = len(custom_model.infected_agents) + len(custom_model.recovered_agents)
    print(f"   总影响用户: {total_influenced}")
    print(f"   传播步数: {len(history)}")
    print(f"   传播比例: {total_influenced/len(users)*100:.1f}%")

    # 按用户类型分析
    type_stats = {}
    all_influenced = custom_model.infected_agents.union(custom_model.recovered_agents)
    for user_id in all_influenced:
        user = manager.get_agent_by_id(user_id)
        if user:
            user_type = user.personality
            type_stats[user_type] = type_stats.get(user_type, 0) + 1

    print("\n   按用户类型分布:")
    for user_type, count in type_stats.items():
        total_of_type = len([u for u in users if u.personality == user_type])
        percentage = count / total_of_type * 100 if total_of_type > 0 else 0
        print(f"     - {user_type}: {count}/{total_of_type} ({percentage:.1f}%)")

    # 7. 传播策略建议
    print("\n💡 营销策略建议:")

    if total_influenced / len(users) > 0.5:
        print("   ✅ 产品传播效果良好！建议:")
        print("      - 继续利用分析型用户进行推广")
        print("      - 扩大传播范围到更多用户群体")
    else:
        print("   ⚠️ 传播效果有限，建议优化:")
        print("      - 增加种子用户的多样性")
        print("      - 调整传播内容以提高接受度")
        print("      - 针对随意型用户群体制定专门策略")

    return history

if __name__ == "__main__":
    simulate_product_viral_marketing()