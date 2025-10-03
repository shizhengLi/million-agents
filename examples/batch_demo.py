#!/usr/bin/env python3
"""
Batch Agent Manager Demo
Demonstrates the batch agent creation and management capabilities
"""

import sys
import os
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.batch_manager import BatchAgentManager


def main():
    """Main demo function"""
    print("=== 百万级智能体批量管理演示 ===\n")

    # Create batch manager
    print("1. 创建批量智能体管理器...")
    manager = BatchAgentManager(max_agents=10000, batch_size=50)
    print(f"   - 最大智能体数量: {manager.max_agents:,}")
    print(f"   - 批处理大小: {manager.batch_size}")

    # Create batch of agents
    print("\n2. 批量创建智能体...")
    start_time = time.time()
    agents = manager.create_batch_agents(
        count=100,
        name_prefix="DemoAgent",
        personalities=["friendly", "analytical", "creative"]
    )
    creation_time = time.time() - start_time
    print(f"   - 创建了 {len(agents)} 个智能体")
    print(f"   - 耗时: {creation_time:.3f} 秒")
    print(f"   - 速度: {len(agents)/creation_time:.1f} 智能体/秒")

    # Create some friendships
    print("\n3. 建立智能体之间的好友关系...")
    friendships = manager.create_random_friendships(max_friends_per_agent=3)
    print(f"   - 创建了 {friendships} 个好友关系")

    # Create communities
    print("\n4. 创建智能体社区...")
    communities = manager.create_communities(
        community_names=["AI_研究者", "数据科学家", "开发者社区"],
        max_members_per_community=40
    )

    for community_name, members in communities.items():
        print(f"   - {community_name}: {len(members)} 个成员")

    # Get statistics
    print("\n5. 智能体群体统计...")
    stats = manager.get_statistics()
    print(f"   - 总智能体数: {stats['total_agents']}")
    print(f"   - 平均好友数: {stats['average_friends']:.1f}")
    print(f"   - 有社区的智能体: {stats['agents_in_communities']}")
    print(f"   - 总社区数: {stats['total_communities']}")

    print("\n   个性分布:")
    for personality, count in stats['personalities'].items():
        print(f"     - {personality}: {count}")

    print("\n   热门兴趣:")
    for interest, count in list(stats['common_interests'].items())[:5]:
        print(f"     - {interest}: {count}")

    # Test filtering
    print("\n6. 智能体筛选功能...")
    friendly_agents = manager.get_agents_by_personality("friendly")
    ai_agents = manager.get_agents_by_interest("AI")

    print(f"   - 友好型智能体: {len(friendly_agents)} 个")
    print(f"   - 对AI感兴趣的智能体: {len(ai_agents)} 个")

    # Run batch interactions
    print("\n7. 批量交互演示...")
    interactions = manager.run_batch_interactions(
        context="讨论人工智能的未来发展",
        max_interactions=5
    )

    print(f"   - 生成了 {len(interactions)} 个交互")
    for i, interaction in enumerate(interactions[:2], 1):
        print(f"   - 交互 {i}:")
        print(f"     智能体: {interaction['agent_name']} ({interaction['personality']})")
        print(f"     消息: {interaction['message'][:100]}...")

    # Memory usage
    print("\n8. 内存使用情况...")
    memory_info = manager.get_memory_usage()
    print(f"   - 智能体数量: {memory_info['total_agents']}")
    print(f"   - 估计内存使用: {memory_info['estimated_memory_mb']:.2f} MB")
    print(f"   - 每个智能体内存: {memory_info['memory_per_agent_kb']:.2f} KB")

    # Performance metrics
    print("\n9. 性能指标...")
    metrics = manager.get_performance_metrics()
    print(f"   - 创建的智能体数: {metrics['agents_created']}")
    print(f"   - 创建时间: {metrics['creation_time_seconds']:.3f} 秒")
    print(f"   - 创建速度: {metrics['agents_per_second']:.1f} 智能体/秒")
    print(f"   - 利用率: {metrics['utilization_percent']:.2f}%")

    # Export data
    print("\n10. 数据导出演示...")
    json_data = manager.export_agents(format='json')
    print(f"    - 导出了 {len(json_data)} 字符的JSON数据")

    # Show a sample agent
    print("\n11. 示例智能体详情...")
    sample_agent = agents[0]
    print(f"    - ID: {sample_agent.agent_id}")
    print(f"    - 姓名: {sample_agent.name}")
    print(f"    - 个性: {sample_agent.personality}")
    print(f"    - 兴趣: {', '.join(sample_agent.interests)}")
    print(f"    - 好友数: {len(sample_agent.friends)}")
    print(f"    - 社区: {', '.join(sample_agent.communities) if sample_agent.communities else '无'}")

    print(f"\n=== 演示完成! 成功展示了批量智能体管理功能 ===")
    print(f"提示: 这个系统可以扩展到支持百万级智能体!")


if __name__ == "__main__":
    main()