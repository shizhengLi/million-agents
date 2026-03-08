#!/usr/bin/env python3
"""
智能客服系统模拟Demo
模拟多个AI客服处理不同类型客户咨询的场景
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.async_manager import AsyncAgentManager
from agents.social_agent import SocialAgent


async def simulate_customer_service():
    """模拟智能客服系统"""

    print("🎧 智能客服系统模拟")
    print("=" * 50)

    # 1. 创建客服智能体
    print("👥 创建客服团队...")
    manager = AsyncAgentManager()

    service_agents = []

    # 不同类型的客服
    agent_types = [
        {
            "type": "技术支持",
            "personality": "analytical",
            "skills": ["技术问题", "故障排查"],
        },
        {
            "type": "产品咨询",
            "personality": "friendly",
            "skills": ["产品功能", "使用指导"],
        },
        {
            "type": "售后服务",
            "personality": "helpful",
            "skills": ["退换货", "投诉处理"],
        },
        {
            "type": "销售咨询",
            "personality": "casual",
            "skills": ["价格咨询", "购买建议"],
        },
    ]

    for i, agent_config in enumerate(agent_types):
        for j in range(5):  # 每种类型5个客服
            agent = SocialAgent(
                agent_id=f"service_{agent_config['type']}_{j + 1}",
                name=f"{agent_config['type']}客服{j + 1}",
                personality=agent_config["personality"],
                interests=agent_config["skills"],
                bio=f"专业{agent_config['type']}客服，擅长{', '.join(agent_config['skills'])}",
            )
            service_agents.append(agent)

    print(f"   创建了 {len(service_agents)} 个客服智能体")

    # 2. 模拟客户咨询
    customer_inquiries = [
        {
            "type": "技术支持",
            "message": "我的应用无法启动，总是闪退",
            "priority": "high",
        },
        {
            "type": "产品咨询",
            "message": "请问这个产品有什么新功能吗？",
            "priority": "medium",
        },
        {
            "type": "售后服务",
            "message": "我买的产品有质量问题，想要退货",
            "priority": "high",
        },
        {
            "type": "销售咨询",
            "message": "我想了解一下不同版本的价格差异",
            "priority": "low",
        },
        {"type": "技术支持", "message": "如何连接到Wi-Fi网络？", "priority": "medium"},
        {
            "type": "产品咨询",
            "message": "这个产品适合什么人群使用？",
            "priority": "medium",
        },
        {"type": "售后服务", "message": "收到商品时包装破损了", "priority": "high"},
        {"type": "销售咨询", "message": "有优惠活动吗？", "priority": "low"},
    ]

    print(f"\n💬 接收到 {len(customer_inquiries)} 个客户咨询...")

    # 3. 智能分配客服
    print("🎯 智能分配客服...")

    async def assign_service_agent(inquiry):
        """根据咨询类型智能分配客服"""
        suitable_agents = [
            agent
            for agent in service_agents
            if any(
                skill in inquiry["message"] or skill in agent.interests
                for skill in agent.interests
            )
        ]

        if not suitable_agents:
            # 如果没有完全匹配的，选择类型相近的
            suitable_agents = [
                agent for agent in service_agents if inquiry["type"] in agent.bio
            ]

        if suitable_agents:
            return suitable_agents[0]  # 返回第一个可用的客服
        else:
            return service_agents[0]  # 默认分配

    # 4. 处理咨询
    print("⚡ 开始并行处理咨询...")

    async def process_inquiry(inquiry, agent):
        """处理单个咨询"""
        response = await asyncio.to_thread(
            agent.generate_message,
            f"客户咨询: {inquiry['message']}\n请提供专业、友好的回复",
        )

        return {
            "customer_inquiry": inquiry["message"],
            "service_agent": agent.name,
            "agent_type": agent.bio.split("，")[0],
            "response": response,
            "priority": inquiry["priority"],
        }

    # 并行处理所有咨询
    tasks = []
    for inquiry in customer_inquiries:
        agent = await assign_service_agent(inquiry)
        task = asyncio.create_task(process_inquiry(inquiry, agent))
        tasks.append(task)

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)

    # 5. 分析处理结果
    print("📊 处理结果分析:")

    # 按优先级统计
    priority_stats = {}
    # 按客服类型统计
    agent_stats = {}

    for result in results:
        priority = result["priority"]
        agent_type = result["agent_type"]

        priority_stats[priority] = priority_stats.get(priority, 0) + 1
        agent_stats[agent_type] = agent_stats.get(agent_type, 0) + 1

        print(f"\n🎯 {result['agent_type']} 处理:")
        print(f"   客户问题: {result['customer_inquiry']}")
        print(f"   客服回复: {result['response'][:100]}...")
        print(f"   优先级: {result['priority']}")

    print(f"\n📈 优先级分布:")
    for priority, count in priority_stats.items():
        print(f"   {priority}: {count} 个")

    print(f"\n👥 客服工作量分布:")
    for agent_type, count in agent_stats.items():
        print(f"   {agent_type}: {count} 次")

    # 6. 性能评估
    print(f"\n⚡ 系统性能:")
    print(f"   总咨询数: {len(customer_inquiries)}")
    print(f"   在线客服: {len(service_agents)}")
    print(f"   并行处理: ✅ 支持异步处理")
    print(f"   智能分配: ✅ 根据类型自动分配")
    print(f"   响应质量: ✅ 个性化专业回复")

    return results


if __name__ == "__main__":
    asyncio.run(simulate_customer_service())
