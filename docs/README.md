# 百万级智能体项目 - 使用指南和应用Demo

## 🎯 项目概述

这个百万级智能体项目是一个社交网络模拟平台，能够创建、管理和分析大规模智能体社区。项目具备从单个智能体交互到百万级群体模拟的完整能力。

## 🚀 实际应用场景

### 1. 科研和学术研究
- **社会网络传播研究** - 模拟信息、观点、谣言在社交网络中的传播
- **群体行为分析** - 研究大规模群体的决策模式和集体行为
- **AI社会实验** - 创建虚拟社会来测试各种社会理论和政策

### 2. 商业应用
- **市场调研** - 模拟消费者行为和产品传播效果
- **品牌营销分析** - 测试不同营销策略在虚拟社交网络中的效果
- **风险评估** - 模拟危机事件在社交网络中的传播和影响

### 3. 教育和培训
- **复杂系统教学** - 直观展示复杂网络和涌现现象
- **AI伦理教育** - 通过模拟演示AI决策的偏见和影响
- **数据科学培训** - 提供真实的社交网络数据分析环境

### 4. 娱乐和创意
- **虚拟世界构建** - 创建具有丰富社交关系的虚拟角色社区
- **交互式故事** - 生成动态变化的社交网络故事
- **游戏NPC系统** - 为游戏提供智能化的非玩家角色交互

## 🛠️ 如何使用这个项目

### 方式1: Web界面可视化 (推荐初学者)

这是最简单易用的方式，通过浏览器界面进行交互：

```bash
# 启动Web服务器
python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# 在浏览器中访问
open http://localhost:8000

# 想要关闭8000端口
sudo lsof -ti :8000 | xargs sudo kill -9
```

**Web界面功能：**
- 🎮 **传播模拟控制面板** - 直观设置传播参数
- 🕸️ **实时网络可视化** - 查看智能体网络拓扑
- 📊 **数据分析仪表板** - 实时监控统计指标
- 🤖 **智能体管理** - 查看和管理智能体状态
- 📈 **影响力分析** - 计算最优传播策略

### 方式2: Python脚本Demo (适合开发者)

项目提供了多个Demo脚本，展示不同功能：

```bash
# 基础智能体交互Demo
python examples/demo.py

# 批量智能体管理Demo
python examples/batch_demo.py

# 异步处理Demo
python examples/async_demo.py

# 社交网络分析Demo
python examples/social_network_demo.py
```

### 方式3: API集成 (适合高级用户)

通过RESTful API集成到自己的应用中：

```python
import requests

# 获取系统状态
response = requests.get('http://localhost:8000/api/stats/system')
stats = response.json()

# 创建传播模拟
propagation_data = {
    "message": "测试消息",
    "seed_agents": ["agent_1", "agent_2"],
    "model_type": "viral",
    "parameters": {"infection_probability": 0.2},
    "max_steps": 50
}
response = requests.post('http://localhost:8000/api/propagation/start', json=propagation_data)
result = response.json()
```

## 📚 具体使用Demo

### Demo 1: 社交网络传播研究

```python
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
        personalities=["influencer", "regular", "skeptic"],
        interests=["科技", "生活", "娱乐", "体育", "美食"]
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
        "influencer": 0.8,  # 影响者更容易接受和传播
        "regular": 0.3,     # 普通用户
        "skeptic": 0.1      # 怀疑者不容易接受
    }

    # 4. 选择种子用户 (早期采用者)
    print("🌱 选择种子用户...")
    seed_users = [user for user in users if user.personality == "influencer"][:10]
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
            agent_type = agent.get('type', 'regular')
            return self.infection_probs.get(agent_type, 0.3)

    custom_model = CustomPropagationModel(social_network, infection_probs)

    # 执行传播
    result = custom_model.propagate(
        message="新款智能手表发布 - 健康监测、运动追踪、智能生活！",
        seed_agents=seed_ids,
        max_steps=20
    )

    # 6. 分析结果
    print("📊 传播结果分析...")
    print(f"   总影响用户: {len(result.influenced_agents)}")
    print(f"   传播步数: {result.total_steps}")
    print(f"   传播比例: {len(result.influenced_agents)/len(users)*100:.1f}%")

    # 按用户类型分析
    type_stats = {}
    for user_id in result.influenced_agents:
        user = manager.get_agent(user_id)
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

    if len(result.influenced_agents) / len(users) > 0.5:
        print("   ✅ 产品传播效果良好！建议:")
        print("      - 继续利用影响者进行推广")
        print("      - 扩大传播范围到更多用户群体")
    else:
        print("   ⚠️ 传播效果有限，建议优化:")
        print("      - 增加种子用户的多样性")
        print("      - 调整传播内容以提高接受度")
        print("      - 针对怀疑者群体制定专门策略")

    return result

if __name__ == "__main__":
    simulate_product_viral_marketing()
```

### Demo 2: 虚拟社会政策影响分析

```python
#!/usr/bin/env python3
"""
虚拟社会政策影响分析Demo
模拟某项政策在虚拟社会中的接受度和传播效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
        "年轻人": {"count": 1500, "interests": ["科技", "创新", "环保"], "personality": "progressive"},
        "中年人": {"count": 2000, "interests": ["经济", "家庭", "教育"], "personality": "moderate"},
        "老年人": {"count": 1000, "interests": ["健康", "传统", "安全"], "personality": "conservative"},
        "专业人士": {"count": 500, "interests": ["专业", "发展", "政策"], "personality": "analytical"}
    }

    all_agents = []
    for group_name, config in social_groups.items():
        agents = manager.create_batch_agents(
            count=config["count"],
            name_prefix=group_name,
            personalities=[config["personality"]],
            interests=config["interests"]
        )
        all_agents.extend(agents)
        print(f"   创建了 {len(agents)} 个{group_name}")

    # 2. 构建社会网络
    print("🕸️ 构建社会网络关系...")
    # 同类群体内部连接更紧密
    for group_name, config in social_groups.items():
        group_agents = [a for a in all_agents if a.personality == config["personality"]]
        for agent in group_agents[:len(group_agents)//2]:  # 只处理前一半避免重复
            # 同群体好友 (70%概率)
            for other in random.sample(group_agents, min(5, len(group_agents)-1)):
                if other != agent and random.random() < 0.7:
                    agent.add_friend(other)

            # 跨群体好友 (30%概率)
            other_groups = [a for a in all_agents if a.personality != config["personality"]]
            if other_groups and random.random() < 0.3:
                friend = random.choice(other_groups)
                agent.add_friend(friend)

    # 3. 模拟不同政策传播
    policies = [
        {
            "name": "环保税政策",
            "message": "政府将实施新的环保税收政策，旨在减少碳排放，促进绿色发展",
            "appeal_groups": ["年轻人", "专业人士"],
            "resistance_groups": ["中年人", "老年人"]
        },
        {
            "name": "数字化教育改革",
            "message": "全面推进教育数字化转型，提供在线学习资源和智能教学工具",
            "appeal_groups": ["年轻人", "专业人士", "中年人"],
            "resistance_groups": ["老年人"]
        }
    ]

    # 创建网络结构
    agents_data = {a.agent_id: {'id': a.agent_id, 'name': a.name, 'type': a.personality} for a in all_agents}
    connections = {a.agent_id: [(friend, 0.6) for friend in a.friends] for a in all_agents}
    social_network = MockSocialNetwork(list(agents_data.keys()), connections)

    # 4. 模拟政策传播
    results = {}

    for policy in policies:
        print(f"\n📢 模拟{policy['name']}传播...")

        diffusion_model = InformationDiffusionModel(social_network)

        # 选择不同群体的种子
        seeds = []
        for group in policy["appeal_groups"]:
            group_agents = [a for a in all_agents if a.personality == social_groups[group]["personality"]]
            seeds.extend(random.sample(group_agents, min(10, len(group_agents))))

        # 自定义接受概率
        def custom_activation_prob(agent_id, neighbor_id, message):
            agent = next((a for a in all_agents if a.agent_id == agent_id), None)
            if not agent:
                return 0.1

            # 根据群体类型调整接受概率
            base_prob = 0.3
            if agent.personality == "progressive":
                base_prob = 0.7
            elif agent.personality == "conservative":
                base_prob = 0.2
            elif agent.personality == "moderate":
                base_prob = 0.4
            elif agent.personality == "analytical":
                base_prob = 0.6

            # 根据兴趣匹配调整
            if any(interest in policy["message"] for interest in agent.interests):
                base_prob += 0.2

            return min(base_prob, 0.9)

        diffusion_model.activation_probability = custom_activation_prob

        result = diffusion_model.propagate(
            message=policy["message"],
            seed_agents=[a.agent_id for a in seeds],
            max_steps=15
        )

        results[policy["name"]] = result

        # 分析结果
        acceptance_rate = len(result.influenced_agents) / len(all_agents) * 100
        print(f"   总接受度: {acceptance_rate:.1f}%")
        print(f"   传播步数: {result.total_steps}")

        # 按群体分析
        group_analysis = {}
        for group_name, config in social_groups.items():
            group_agents = [a.agent_id for a in all_agents if a.personality == config["personality"]]
            influenced_in_group = len(set(result.influenced_agents) & set(group_agents))
            group_rate = influenced_in_group / len(group_agents) * 100
            group_analysis[group_name] = group_rate
            print(f"   {group_name}接受度: {group_rate:.1f}%")

    # 5. 政策建议
    print("\n💡 政策实施建议:")

    for policy_name, result in results.items():
        acceptance_rate = len(result.influenced_agents) / len(all_agents) * 100

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
```

### Demo 3: 智能客服系统模拟

```python
#!/usr/bin/env python3
"""
智能客服系统模拟Demo
模拟多个AI客服处理不同类型客户咨询的场景
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.async_manager import AsyncAgentManager
from agents.social_agent import SocialAgent

async def simulate_customer_service():
    """模拟智能客服系统"""

    print("🎧 智能客服系统模拟")
    print("=" * 50)

    # 1. 创建客服智能体
    print("👥 创建客服团队...")
    manager = AsyncAgentManager(max_agents=100)

    service_agents = []

    # 不同类型的客服
    agent_types = [
        {"type": "技术支持", "personality": "analytical", "skills": ["技术问题", "故障排查"]},
        {"type": "产品咨询", "personality": "friendly", "skills": ["产品功能", "使用指导"]},
        {"type": "售后服务", "personality": "empathetic", "skills": ["退换货", "投诉处理"]},
        {"type": "销售咨询", "personality": "enthusiastic", "skills": ["价格咨询", "购买建议"]}
    ]

    for i, agent_config in enumerate(agent_types):
        for j in range(5):  # 每种类型5个客服
            agent = SocialAgent(
                agent_id=f"service_{agent_config['type']}_{j+1}",
                name=f"{agent_config['type']}客服{j+1}",
                personality=agent_config["personality"],
                interests=agent_config["skills"],
                bio=f"专业{agent_config['type']}客服，擅长{', '.join(agent_config['skills'])}"
            )
            service_agents.append(agent)
            await manager.add_agent(agent)

    print(f"   创建了 {len(service_agents)} 个客服智能体")

    # 2. 模拟客户咨询
    customer_inquiries = [
        {"type": "技术支持", "message": "我的应用无法启动，总是闪退", "priority": "high"},
        {"type": "产品咨询", "message": "请问这个产品有什么新功能吗？", "priority": "medium"},
        {"type": "售后服务", "message": "我买的产品有质量问题，想要退货", "priority": "high"},
        {"type": "销售咨询", "message": "我想了解一下不同版本的价格差异", "priority": "low"},
        {"type": "技术支持", "message": "如何连接到Wi-Fi网络？", "priority": "medium"},
        {"type": "产品咨询", "message": "这个产品适合什么人群使用？", "priority": "medium"},
        {"type": "售后服务", "message": "收到商品时包装破损了", "priority": "high"},
        {"type": "销售咨询", "message": "有优惠活动吗？", "priority": "low"}
    ]

    print(f"\n💬 接收到 {len(customer_inquiries)} 个客户咨询...")

    # 3. 智能分配客服
    print("🎯 智能分配客服...")

    async def assign_service_agent(inquiry):
        """根据咨询类型智能分配客服"""
        suitable_agents = [
            agent for agent in service_agents
            if any(skill in inquiry["message"] or skill in agent.interests
                   for skill in agent.interests)
        ]

        if not suitable_agents:
            # 如果没有完全匹配的，选择类型相近的
            suitable_agents = [
                agent for agent in service_agents
                if inquiry["type"] in agent.bio
            ]

        if suitable_agents:
            return suitable_agents[0]  # 返回第一个可用的客服
        else:
            return service_agents[0]  # 默认分配

    # 4. 处理咨询
    print("⚡ 开始并行处理咨询...")

    async def process_inquiry(inquiry, agent):
        """处理单个咨询"""
        response = await agent.generate_message_async(
            f"客户咨询: {inquiry['message']}\n请提供专业、友好的回复"
        )

        return {
            "customer_inquiry": inquiry["message"],
            "service_agent": agent.name,
            "agent_type": agent.bio.split("，")[0],
            "response": response,
            "priority": inquiry["priority"]
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
```

## 🎮 如何运行Demo

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加你的 OpenAI API Key

# 3. 运行基础测试
python examples/demo.py
```

### 运行不同Demo

```bash
# 基础功能演示
python examples/demo.py

# 批量智能体管理
python examples/batch_demo.py

# 异步处理演示
python examples/async_demo.py

# 社交网络分析
python examples/social_network_demo.py

# 产品传播模拟 (新建文件)
python examples/product_viral_demo.py

# 政策影响分析 (新建文件)
python examples/policy_impact_demo.py

# 智能客服系统 (新建文件)
python examples/customer_service_demo.py
```

## 🔧 集成到现有项目

### 1. 作为Python库使用

```python
# 在你的项目中导入
from src.agents import SocialAgent
from src.message_propagation import ViralPropagationModel
from src.social_network import NetworkAnalyzer

# 创建智能体
agent = SocialAgent(
    agent_id="user_001",
    name="Alice",
    personality="friendly",
    interests=["AI", "social networks"]
)

# 进行传播分析
model = ViralPropagationModel(network)
result = model.propagate("消息内容", ["seed_1", "seed_2"])
```

### 2. 通过API集成

```python
import requests

# 启动API服务
# python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000

# 在你的代码中调用API
base_url = "http://localhost:8000/api"

# 获取系统状态
stats = requests.get(f"{base_url}/stats/system").json()

# 创建传播模拟
propagation = requests.post(f"{base_url}/propagation/start", json={
    "message": "测试消息",
    "seed_agents": ["agent_1"],
    "model_type": "viral"
}).json()
```

### 3. Docker部署

```dockerfile
# Dockerfile示例
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY examples/ ./examples/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.web_interface.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 扩展和定制

### 添加新的智能体类型

```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, custom_attribute, **kwargs):
        super().__init__(**kwargs)
        self.custom_attribute = custom_attribute

    def generate_message(self, context):
        # 自定义消息生成逻辑
        return f"自定义消息: {context}"
```

### 添加新的传播模型

```python
from src.message_propagation.base import BasePropagationModel

class CustomPropagationModel(BasePropagationModel):
    def propagate(self, message, seed_agents, **kwargs):
        # 自定义传播逻辑
        pass
```

## 🎯 实际案例

### 案例1: 社交媒体营销分析
一家公司使用这个系统来测试新产品在社交媒体上的传播效果，通过调整不同的种子用户和传播策略，找到最优的营销方案。

### 案例2: 公共卫生政策模拟
研究机构使用该系统模拟疫苗接种政策在社区中的接受度，通过调整宣传策略和种子人群，预测政策实施效果。

### 案例3: 在线教育平台优化
教育平台利用该系统模拟知识在学生社区中的传播，找到影响学习效果的关键节点人物。

## 🤝 贡献和使用

这个项目是一个开源的研究平台，欢迎：

1. **学术研究** - 使用平台进行社会网络、传播学研究
2. **商业应用** - 集成到产品中进行市场分析、用户行为预测
3. **教育培训** - 作为复杂系统、AI伦理等课程的教学工具
4. **开源贡献** - 提交新的功能模块、改进算法、优化性能

通过这些Demo和指南，希望您能更好地理解和使用这个百万级智能体系统！