#!/usr/bin/env python3
"""
百万智能体项目 - 快速开始Demo
展示项目的主要功能和使用方法
"""

import sys
import os
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_basic_agent_creation():
    """演示基础智能体创建"""
    print("🤖 Demo 1: 基础智能体创建")
    print("-" * 40)

    try:
        from agents.social_agent import SocialAgent

        # 创建几个不同类型的智能体
        alice = SocialAgent(
            agent_id="alice_001",
            name="Alice",
            personality="friendly",
            interests=["AI", "机器学习", "社交网络"],
            bio="AI研究员，对社交动态感兴趣"
        )

        bob = SocialAgent(
            agent_id="bob_002",
            name="Bob",
            personality="analytical",
            interests=["数据科学", "统计", "研究"],
            bio="数据科学家，具有分析思维"
        )

        print(f"✅ 创建了智能体: {alice.name} ({alice.personality})")
        print(f"✅ 创建了智能体: {bob.name} ({bob.personality})")

        # 测试兼容性
        compatibility = alice.check_compatibility(bob)
        print(f"💝 兼容性评分: {compatibility:.2%}")

        # 建立好友关系
        if compatibility > 0.4:
            alice.add_friend(bob)
            print(f"🤝 {alice.name} 和 {bob.name} 成为好友！")

        # 生成消息
        message = alice.generate_message("你好，很高兴认识你！")
        print(f"💬 {alice.name}: {message}")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已正确安装所有依赖")
        return False

    print("✅ Demo 1 完成!\n")
    return True

def demo_batch_management():
    """演示批量智能体管理"""
    print("👥 Demo 2: 批量智能体管理")
    print("-" * 40)

    try:
        from agents.batch_manager import BatchAgentManager

        # 创建批量管理器
        manager = BatchAgentManager(max_agents=1000, batch_size=50)
        print(f"✅ 创建批量管理器 (最大: {manager.max_agents:,})")

        # 批量创建智能体
        start_time = time.time()
        agents = manager.create_batch_agents(
            count=100,
            name_prefix="User",
            personalities=["friendly", "analytical", "creative"],
            interests_list=[["科技", "艺术", "运动", "音乐", "阅读"]]
        )
        creation_time = time.time() - start_time

        print(f"✅ 批量创建 {len(agents)} 个智能体")
        print(f"⚡ 创建速度: {len(agents)/creation_time:.1f} 智能体/秒")

        # 创建随机好友关系
        friendships = manager.create_random_friendships(max_friends_per_agent=5)
        print(f"🤝 创建了 {friendships} 个好友关系")

        # 获取统计信息
        stats = manager.get_statistics()
        print(f"📊 平均好友数: {stats['average_friends']:.1f}")
        print(f"🏘️ 有社区的智能体: {stats['agents_in_communities']}")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False

    print("✅ Demo 2 完成!\n")
    return True

def demo_web_interface():
    """演示Web界面功能"""
    print("🌐 Demo 3: Web界面功能")
    print("-" * 40)

    try:
        import requests
        import json

        # 检查Web服务器是否运行
        base_url = "http://localhost:8000/api"

        try:
            response = requests.get(f"{base_url}/stats/system", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print("✅ Web服务器运行正常")
                print(f"🤖 活跃智能体: {stats['agents']['active_agents']}")
                print(f"🔗 网络连接数: {stats['social_network']['total_connections']}")
                print(f"📊 平均声誉: {stats['reputation_system']['average_reputation']:.1f}")

                # 测试传播API
                propagation_data = {
                    "message": "测试消息传播",
                    "seed_agents": ["agent_1"],
                    "model_type": "viral",
                    "parameters": {"infection_probability": 0.2},
                    "max_steps": 10
                }

                prop_response = requests.post(f"{base_url}/propagation/start", json=propagation_data, timeout=10)
                if prop_response.status_code == 200:
                    result = prop_response.json()
                    print(f"📢 传播模拟成功: 影响了 {len(result['influenced_agents'])} 个智能体")
                else:
                    print("⚠️ 传播API调用失败")

                print("🌐 Web界面访问: http://localhost:8000")

            else:
                print("⚠️ Web服务器响应异常")

        except requests.exceptions.ConnectionError:
            print("⚠️ Web服务器未运行")
            print("💡 启动命令: python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload")
            return False
        except requests.exceptions.Timeout:
            print("⚠️ Web服务器响应超时")
            return False

    except ImportError:
        print("⚠️ requests库未安装，跳过Web测试")
        print("💡 安装命令: pip install requests")
        return False

    print("✅ Demo 3 完成!\n")
    return True

def demo_propagation_models():
    """演示传播模型"""
    print("📈 Demo 4: 消息传播模型")
    print("-" * 40)

    try:
        from message_propagation.viral_propagation import ViralPropagationModel
        from message_propagation.social_network import MockSocialNetwork
        from agents.batch_manager import BatchAgentManager

        # 创建小型网络进行演示
        manager = BatchAgentManager(max_agents=50, batch_size=10)
        agents = manager.create_batch_agents(count=20, name_prefix="Node")
        manager.create_random_friendships(max_friends_per_agent=3)

        # 构建网络结构
        agents_data = {a.agent_id: {'id': a.agent_id, 'name': a.name, 'type': a.personality} for a in agents}
        connections = {a.agent_id: [(friend, 0.7) for friend in a.friends] for a in agents}
        network = MockSocialNetwork(list(agents_data.keys()), connections)

        # 创建传播模型
        model = ViralPropagationModel(network)
        print("✅ 创建病毒式传播模型")

        # 执行传播
        seed_agents = [agents[0].agent_id, agents[1].agent_id]

        # 设置传播参数
        model.message = "重要通知：系统将在今晚进行维护升级"
        model.seed_agents = seed_agents
        model.max_steps = 5

        # 设置种子智能体并执行传播模拟
        model.set_initial_infected(seed_agents)
        history = model.propagate_full_simulation()

        if history:
            final_step = history[-1]
            total_influenced = len(model.infected_agents) + len(model.recovered_agents)

            print(f"📢 传播结果:")
            print(f"   - 种子智能体: {len(seed_agents)}")
            print(f"   - 影响智能体: {total_influenced}")
            print(f"   - 传播步数: {len(history)}")
            print(f"   - 传播比例: {total_influenced/len(agents)*100:.1f}%")
        else:
            print("📢 传播结果: 未发生传播")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False

    print("✅ Demo 4 完成!\n")
    return True

def main():
    """主演示函数"""
    print("🚀 百万智能体项目 - 快速开始演示")
    print("=" * 60)
    print()

    demos = [
        ("基础智能体创建", demo_basic_agent_creation),
        ("批量智能体管理", demo_batch_management),
        ("Web界面功能", demo_web_interface),
        ("传播模型演示", demo_propagation_models)
    ]

    success_count = 0

    for demo_name, demo_func in demos:
        print(f"🎯 正在运行: {demo_name}")
        if demo_func():
            success_count += 1
        print()

    # 总结
    print("=" * 60)
    print(f"📊 演示完成: {success_count}/{len(demos)} 个成功")

    if success_count == len(demos):
        print("🎉 所有功能都正常工作！")
        print("\n💡 下一步建议:")
        print("   1. 访问 Web 界面: http://localhost:8000")
        print("   2. 查看完整文档: docs/web-interface-visualization-guide.md")
        print("   3. 运行其他Demo: python examples/batch_demo.py")
        print("   4. 开始你的百万智能体实验！")
    else:
        print("⚠️ 部分功能需要配置")
        print("\n🔧 故障排除:")
        print("   1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("   2. 检查环境变量配置 (.env 文件)")
        print("   3. 启动Web服务器进行完整功能测试")

    print(f"\n📚 更多信息请查看: docs/usage-guide-and-demos.md")

if __name__ == "__main__":
    main()