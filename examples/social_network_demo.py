#!/usr/bin/env python3
"""
社交网络算法演示
展示如何使用社交网络模块进行各种分析和可视化
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.social_network import (
    SocialNetworkGraph,
    PageRankCalculator,
    CommunityDetector,
    ShortestPathCalculator,
    SocialNetworkVisualizer
)


def create_demo_network():
    """创建演示用的社交网络"""
    graph = SocialNetworkGraph()

    # 添加Agent
    agents = [
        (1, "Alice"), (2, "Bob"), (3, "Charlie"), (4, "Diana"),
        (5, "Eve"), (6, "Frank"), (7, "Grace"), (8, "Henry"),
        (9, "Ivy"), (10, "Jack")
    ]

    for agent_id, name in agents:
        graph.add_agent(agent_id, name)

    # 创建社区1：Alice, Bob, Charlie (紧密连接)
    graph.add_friendship(1, 2, strength=0.9)
    graph.add_friendship(1, 3, strength=0.8)
    graph.add_friendship(2, 3, strength=0.9)

    # 创建社区2：Diana, Eve, Frank (紧密连接)
    graph.add_friendship(4, 5, strength=0.9)
    graph.add_friendship(4, 6, strength=0.7)
    graph.add_friendship(5, 6, strength=0.8)

    # 创建社区3：Grace, Henry (紧密连接)
    graph.add_friendship(7, 8, strength=0.9)

    # 添加社区间连接
    graph.add_friendship(3, 4, strength=0.3)  # 社区1到社区2
    graph.add_friendship(6, 7, strength=0.2)  # 社区2到社区3

    # 添加一些中心节点
    graph.add_friendship(1, 9, strength=0.6)  # Ivy连接到Alice
    graph.add_friendship(9, 10, strength=0.5)  # Jack连接到Ivy
    graph.add_friendship(2, 10, strength=0.4)  # Jack连接到Bob

    return graph


def demo_pagerank_analysis(graph):
    """演示PageRank影响力分析"""
    print("=" * 50)
    print("PageRank 影响力分析")
    print("=" * 50)

    calculator = PageRankCalculator()
    rankings = calculator.calculate_pagerank(graph)

    # 显示所有Agent的PageRank分数
    print("\n所有Agent的PageRank分数:")
    for agent_id, score in sorted(rankings.items(), key=lambda x: x[1], reverse=True):
        agent = graph.get_agent_by_id(agent_id)
        print(f"  {agent.name}: {score:.4f}")

    # 显示前5个最有影响力的Agent
    top_influential = calculator.get_top_influential_agents(graph, top_k=5)
    print(f"\n前5个最有影响力的Agent:")
    for rank, (agent_id, score) in enumerate(top_influential, 1):
        agent = graph.get_agent_by_id(agent_id)
        print(f"  {rank}. {agent.name}: {score:.4f}")


def demo_community_detection(graph):
    """演示社区发现"""
    print("\n" + "=" * 50)
    print("社区发现分析")
    print("=" * 50)

    detector = CommunityDetector()
    communities = detector.detect_communities(graph)

    print(f"\n检测到 {len(communities)} 个社区:")
    for i, community in enumerate(communities, 1):
        agent_names = [graph.get_agent_by_id(agent_id).name for agent_id in community]
        print(f"  社区 {i}: {', '.join(agent_names)}")

    # 显示社区统计信息
    stats = detector.get_community_statistics(graph, communities)
    print(f"\n社区统计信息:")
    print(f"  社区数量: {stats['num_communities']}")
    print(f"  平均社区大小: {stats['average_community_size']:.2f}")
    print(f"  最大社区大小: {stats['largest_community_size']}")
    print(f"  最小社区大小: {stats['smallest_community_size']}")


def demo_shortest_path(graph):
    """演示最短路径分析"""
    print("\n" + "=" * 50)
    print("最短路径分析")
    print("=" * 50)

    calculator = ShortestPathCalculator()

    # 计算一些示例路径
    path_pairs = [
        (1, 10),  # Alice到Jack
        (4, 8),   # Diana到Henry
        (7, 2),   # Grace到Bob
    ]

    for start, end in path_pairs:
        start_agent = graph.get_agent_by_id(start)
        end_agent = graph.get_agent_by_id(end)

        # 无权重路径
        path = calculator.calculate_shortest_path(graph, start, end)
        length = calculator.get_path_length(path)

        # 加权路径
        weighted_path = calculator.calculate_shortest_path(
            graph, start, end, use_weights=True
        )
        weight = calculator.get_path_weight(graph, weighted_path)

        print(f"\n{start_agent.name} 到 {end_agent.name}:")
        if path:
            path_names = [graph.get_agent_by_id(node).name for node in path]
            print(f"  最短路径: {' -> '.join(path_names)} (长度: {length})")
            if weighted_path:
                weighted_names = [graph.get_agent_by_id(node).name for node in weighted_path]
                print(f"  加权路径: {' -> '.join(weighted_names)} (权重: {weight:.2f})")
        else:
            print("  无可用路径")

    # 显示网络统计
    avg_length = calculator.calculate_average_path_length(graph)
    diameter = calculator.get_diameter(graph)
    print(f"\n网络统计:")
    print(f"  平均路径长度: {avg_length:.2f}")
    print(f"  网络直径: {diameter}")


def demo_visualization(graph):
    """演示可视化功能"""
    print("\n" + "=" * 50)
    print("可视化演示")
    print("=" * 50)

    # 注意：在无GUI环境中，这些图像会被保存而不是显示
    visualizer = SocialNetworkVisualizer()

    print("生成可视化图像...")

    try:
        # 创建保存目录
        os.makedirs('visualizations', exist_ok=True)

        # 基础网络图
        visualizer.plot_basic_graph(
            graph,
            save_path='visualizations/basic_network.png'
        )
        print("  ✓ 基础网络图已保存")

        # PageRank影响力图
        visualizer.plot_with_pagerank(
            graph,
            save_path='visualizations/pagerank_network.png'
        )
        print("  ✓ PageRank影响力图已保存")

        # 社区图
        visualizer.plot_with_communities(
            graph,
            save_path='visualizations/community_network.png'
        )
        print("  ✓ 社区图已保存")

        # 权重图
        visualizer.plot_weighted_graph(
            graph,
            save_path='visualizations/weighted_network.png'
        )
        print("  ✓ 权重图已保存")

        # 度数分布
        visualizer.plot_degree_distribution(
            graph,
            save_path='visualizations/degree_distribution.png'
        )
        print("  ✓ 度数分布图已保存")

        # PageRank分布
        visualizer.plot_pagerank_distribution(
            graph,
            save_path='visualizations/pagerank_distribution.png'
        )
        print("  ✓ PageRank分布图已保存")

        # 分析仪表板
        visualizer.create_summary_dashboard(
            graph,
            save_path='visualizations/analysis_dashboard.png'
        )
        print("  ✓ 分析仪表板已保存")

        print("\n所有图像已保存到 'visualizations/' 目录")

    except Exception as e:
        print(f"  ⚠️ 可视化生成失败: {e}")
        print("  提示：确保安装了 matplotlib 和 networkx")


def main():
    """主演示函数"""
    print("🎯 社交网络算法演示")
    print("本演示将展示社交网络分析的各种功能")

    # 创建演示网络
    print("\n📊 创建演示社交网络...")
    graph = create_demo_network()

    print(f"创建了包含 {graph.get_agent_count()} 个Agent和 {graph.get_edge_count()} 条关系的网络")

    # 运行各种分析
    demo_pagerank_analysis(graph)
    demo_community_detection(graph)
    demo_shortest_path(graph)
    demo_visualization(graph)

    print("\n" + "=" * 50)
    print("🎉 演示完成！")
    print("=" * 50)
    print("社交网络模块功能总结:")
    print("✓ 社交网络图数据结构")
    print("✓ PageRank影响力排名算法")
    print("✓ 社区发现算法 (Louvain方法)")
    print("✓ 最短路径算法 (BFS + Dijkstra)")
    print("✓ 丰富的可视化功能")
    print("✓ 完整的测试覆盖 (52个测试用例)")


if __name__ == "__main__":
    main()