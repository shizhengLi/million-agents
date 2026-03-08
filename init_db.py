"""
初始化数据库并添加测试数据
"""

import os
import sys
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.database.session import get_session, create_tables
from src.database.agent import Agent
from src.database.friendship import Friendship


def init_database():
    """初始化数据库并添加测试数据"""
    print("创建数据库表...")
    create_tables()

    session = get_session()

    # 检查是否已有数据
    existing_agents = session.query(Agent).count()
    if existing_agents > 0:
        print(f"数据库中已有 {existing_agents} 个智能体")
        response = input("是否清空并重新创建测试数据? (y/n): ")
        if response.lower() != "y":
            return

        # 清空数据
        session.query(Friendship).delete()
        session.query(Agent).delete()
        session.commit()
        print("已清空现有数据")

    # 添加智能体
    print("添加智能体...")
    personality_types = [
        "balanced",
        "explorer",
        "builder",
        "connector",
        "leader",
        "innovator",
    ]
    agents = []

    for i in range(1, 11):
        agent = Agent(
            name=f"Agent_{i}",
            personality_type=random.choice(personality_types),
            openness=random.uniform(0.3, 0.9),
            conscientiousness=random.uniform(0.3, 0.9),
            extraversion=random.uniform(0.3, 0.9),
            agreeableness=random.uniform(0.3, 0.9),
            neuroticism=random.uniform(0.1, 0.7),
        )
        session.add(agent)
        agents.append(agent)

    session.commit()
    print(f"已添加 {len(agents)} 个智能体")

    # 添加朋友关系
    print("添加社交关系...")
    friendship_count = 0
    for i, agent in enumerate(agents):
        # 每个智能体随机连接2-5个其他智能体
        num_connections = random.randint(2, 5)
        potential_targets = [a for a in agents if a.id != agent.id]
        targets = random.sample(
            potential_targets, min(num_connections, len(potential_targets))
        )

        for target in targets:
            # 检查是否已存在关系
            existing = (
                session.query(Friendship)
                .filter(
                    (
                        (Friendship.initiator_id == agent.id)
                        & (Friendship.recipient_id == target.id)
                    )
                    | (
                        (Friendship.initiator_id == target.id)
                        & (Friendship.recipient_id == agent.id)
                    )
                )
                .first()
            )

            if not existing:
                friendship = Friendship(
                    initiator_id=agent.id,
                    recipient_id=target.id,
                    friendship_status="accepted",
                    strength_level=random.uniform(0.3, 1.0),
                    interaction_count=random.randint(0, 50),
                    last_interaction=datetime.utcnow()
                    - timedelta(days=random.randint(0, 30)),
                )
                session.add(friendship)
                friendship_count += 1

    session.commit()
    print(f"已添加 {friendship_count} 个社交关系")

    # 显示创建的数据
    print("\n创建完成的智能体:")
    all_agents = session.query(Agent).all()
    for a in all_agents:
        print(f"  - agent_{a.id}: {a.name} ({a.personality_type})")

    print("\n数据库初始化完成!")
    print(f"\n你可以在Web界面中使用以下种子智能体ID: agent_1, agent_2, ...")

    session.close()


if __name__ == "__main__":
    init_database()
