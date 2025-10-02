#!/usr/bin/env python3
"""
Async Agent Management Demo
Demonstrates high-performance async agent operations for million-scale populations
"""

import sys
import os
import time
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.async_manager import AsyncAgentManager, AsyncConfig
from agents.async_social_agent import AsyncSocialAgent


async def demo_async_agent_creation():
    """Demonstrate async agent creation performance"""
    print("=== 异步智能体创建演示 ===\n")

    # Create async manager with optimized configuration
    config = AsyncConfig(
        max_concurrent=20,
        batch_size=100,
        auto_optimize=True,
        rate_limit_delay=0.01
    )
    manager = AsyncAgentManager(config=config)

    print("1. 创建异步管理器...")
    print(f"   - 最大并发数: {manager.config.max_concurrent}")
    print(f"   - 批处理大小: {manager.config.batch_size}")
    print(f"   - 自动优化: {manager.config.auto_optimize}")

    # Create agents asynchronously
    print("\n2. 异步创建智能体...")
    start_time = time.time()
    agents = await manager.create_agents_async(
        count=500,
        name_prefix="AsyncDemo",
        personalities=["creative", "analytical", "friendly", "formal"]
    )
    creation_time = time.time() - start_time

    print(f"   - 创建了 {len(agents)} 个智能体")
    print(f"   - 耗时: {creation_time:.3f} 秒")
    print(f"   - 速度: {len(agents)/creation_time:.1f} 智能体/秒")
    print(f"   - 平均每智能体: {creation_time/len(agents)*1000:.2f} 毫秒")

    # Show batch size optimization
    final_batch_size = manager.get_current_batch_size()
    print(f"   - 优化后批大小: {final_batch_size}")

    return agents


async def demo_async_social_interactions(agents):
    """Demonstrate async social interactions"""
    print("\n=== 异步社交交互演示 ===\n")

    manager = AsyncAgentManager()

    # Build friendships asynchronously
    print("1. 异步建立好友关系...")
    start_time = time.time()
    friendships = await manager.build_friendships_async(
        agents=agents[:100],
        max_friends_per_agent=5,
        batch_size=20
    )
    friendship_time = time.time() - start_time

    print(f"   - 创建了 {friendships} 个好友关系")
    print(f"   - 耗时: {friendship_time:.3f} 秒")

    # Create communities asynchronously
    print("\n2. 异步创建社区...")
    start_time = time.time()
    communities = await manager.create_communities_async(
        agents=agents[:200],
        community_names=["技术创新", "研究学者", "开发者社区", "设计师群体"],
        max_members_per_community=60
    )
    community_time = time.time() - start_time

    print(f"   - 创建了 {len(communities)} 个社区")
    for community_name, members in communities.items():
        print(f"     - {community_name}: {len(members)} 个成员")
    print(f"   - 耗时: {community_time:.3f} 秒")

    return communities


async def demo_async_message_generation(agents):
    """Demonstrate async message generation with rate limiting"""
    print("\n=== 异步消息生成演示 ===\n")

    # Create async social agents
    async_agents = []
    for agent in agents[:20]:
        async_agent = AsyncSocialAgent(
            name=agent.name,
            personality=agent.personality,
            interests=agent.interests
        )
        async_agents.append(async_agent)

    # Mock OpenAI responses for demo
    import unittest.mock
    with unittest.mock.patch('agents.async_social_agent.AsyncOpenAI') as mock_openai:
        mock_client = unittest.mock.AsyncMock()
        mock_openai.return_value = mock_client

        # Configure mock responses
        mock_response = unittest.mock.Mock()
        mock_response.choices = [unittest.mock.Mock(
            message=unittest.mock.Mock(content="这是一个关于AI和创新的精彩讨论！")
        )]
        mock_response.usage = unittest.mock.Mock(total_tokens=25)
        mock_client.chat.completions.create.return_value = mock_response

        print("1. 批量异步消息生成...")
        start_time = time.time()

        main_agent = async_agents[0]
        interactions = await main_agent.batch_interact_async(
            agents=async_agents[1:10],
            context="讨论人工智能的未来发展趋势",
            max_concurrent=5
        )

        generation_time = time.time() - start_time

        print(f"   - 生成了 {len(interactions)} 个交互")
        print(f"   - 耗时: {generation_time:.3f} 秒")
        print(f"   - 平均速度: {len(interactions)/generation_time:.1f} 交互/秒")

        # Show sample interaction
        if interactions:
            sample = interactions[0]
            agent_id = list(sample.keys())[0]
            interaction = sample[agent_id]
            print(f"\n2. 示例交互:")
            print(f"   - 智能体: {interaction.agent_id}")
            print(f"   - 消息: {interaction.message[:50]}...")
            print(f"   - 处理时间: {interaction.processing_time:.3f} 秒")


async def demo_async_performance_monitoring():
    """Demonstrate async performance monitoring and optimization"""
    print("\n=== 异步性能监控演示 ===\n")

    config = AsyncConfig(
        max_concurrent=15,
        batch_size=50,
        auto_optimize=True
    )
    manager = AsyncAgentManager(config=config)

    # Create agents for performance testing
    print("1. 创建测试智能体...")
    agents = await manager.create_agents_async(count=200, name_prefix="PerfTest")

    # Perform various operations to collect metrics
    print("2. 执行性能测试...")
    await manager.build_friendships_async(agents, max_friends_per_agent=3)
    await manager.create_communities_async(
        agents,
        ["测试社区1", "测试社区2"],
        max_members_per_community=100
    )

    # Get performance metrics
    metrics = manager.get_performance_metrics()

    print("\n3. 性能指标:")
    print(f"   - 总操作数: {metrics['total_operations']}")
    print(f"   - 成功操作: {metrics['successful_operations']}")
    print(f"   - 失败操作: {metrics['failed_operations']}")
    print(f"   - 总耗时: {metrics['total_time']:.3f} 秒")
    print(f"   - 平均操作时间: {metrics['average_operation_time']*1000:.2f} 毫秒")
    print(f"   - 峰值并发: {metrics['peak_concurrent_operations']}")
    print(f"   - 操作速度: {metrics['operations_per_second']:.1f} 操作/秒")
    print(f"   - 当前批大小: {metrics['current_batch_size']}")


async def demo_async_streaming_interactions():
    """Demonstrate streaming async interactions for real-time processing"""
    print("\n=== 异步流式交互演示 ===\n")

    manager = AsyncAgentManager()
    agents = await manager.create_agents_async(
        count=30,
        name_prefix="StreamAgent",
        personalities=["creative", "analytical"]
    )

    # Mock OpenAI for streaming demo
    import unittest.mock
    with unittest.mock.patch('src.agents.async_manager.SocialAgent.generate_message') as mock_generate:
        mock_generate.side_effect = [
            f"流式响应 {i+1}: 这是一个实时的异步响应消息。"
            for i in range(30)
        ]

        print("1. 启动流式交互处理...")
        batch_count = 0
        total_interactions = 0

        start_time = time.time()
        async for batch in manager.stream_interactions(
            agents=agents,
            context="实时流式对话测试",
            batch_size=5
        ):
            batch_count += 1
            total_interactions += len(batch)

            print(f"   批次 {batch_count}: 收到 {len(batch)} 个交互")

            # Show sample from batch
            if batch:
                sample = batch[0]
                print(f"     示例: {sample['message'][:30]}...")

            # Small delay to show streaming effect
            await asyncio.sleep(0.1)

        stream_time = time.time() - start_time

        print(f"\n2. 流式处理完成:")
        print(f"   - 总批次数: {batch_count}")
        print(f"   - 总交互数: {total_interactions}")
        print(f"   - 总耗时: {stream_time:.3f} 秒")
        print(f"   - 吞吐量: {total_interactions/stream_time:.1f} 交互/秒")


async def main():
    """Main demo function"""
    print("=== 百万级智能体异步处理演示 ===\n")
    print("展示高性能异步架构如何支持大规模智能体操作\n")

    try:
        # Demo 1: Async Agent Creation
        agents = await demo_async_agent_creation()

        # Demo 2: Async Social Interactions
        communities = await demo_async_social_interactions(agents)

        # Demo 3: Async Message Generation
        await demo_async_message_generation(agents)

        # Demo 4: Async Performance Monitoring
        await demo_async_performance_monitoring()

        # Demo 5: Async Streaming Interactions
        await demo_async_streaming_interactions()

        print("\n=== 演示完成! ===")
        print("异步处理框架成功展示了:")
        print("✅ 高并发智能体创建")
        print("✅ 异步社交网络构建")
        print("✅ 流式消息生成")
        print("✅ 实时性能监控")
        print("✅ 动态批处理优化")
        print("✅ 流式交互处理")
        print("\n这个异步架构可以扩展到支持百万级智能体的实时交互!")

    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())