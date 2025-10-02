#!/usr/bin/env python3
"""
Demo script showing basic social agent functionality
"""

import os
import sys

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_path)

from src.agents import SocialAgent
from src.config import Settings


def main():
    """Main demo function"""
    print("🤖 Million Agents Social Application Demo")
    print("=" * 50)

    try:
        # Load settings
        settings = Settings()
        print(f"✅ Configuration loaded successfully")
        print(f"   Max agents: {settings.max_agents:,}")
        print(f"   Batch size: {settings.agent_batch_size}")
        print(f"   OpenAI model: {settings.openai_model}")
        print()

        # Create some social agents with different personalities
        print("👥 Creating social agents...")

        alice = SocialAgent(
            agent_id="alice_001",
            name="Alice",
            personality="friendly",
            interests=["AI", "machine learning", "social networks"],
            bio="AI researcher interested in social dynamics"
        )

        bob = SocialAgent(
            agent_id="bob_002",
            name="Bob",
            personality="analytical",
            interests=["data science", "statistics", "research"],
            bio="Data scientist with analytical mindset"
        )

        charlie = SocialAgent(
            agent_id="charlie_003",
            name="Charlie",
            personality="creative",
            interests=["art", "design", "innovation"],
            bio="Creative thinker exploring AI art"
        )

        agents = [alice, bob, charlie]

        # Show agent info
        for agent in agents:
            print(f"   🎭 {agent.name} ({agent.personality})")
            print(f"      Interests: {', '.join(agent.interests)}")
            print(f"      Bio: {agent.bio}")
        print()

        # Test compatibility
        print("💝 Checking agent compatibility...")
        alice_bob_compat = alice.check_compatibility(bob)
        alice_charlie_compat = alice.check_compatibility(charlie)
        bob_charlie_compat = bob.check_compatibility(charlie)

        print(f"   Alice & Bob: {alice_bob_compat:.2%}")
        print(f"   Alice & Charlie: {alice_charlie_compat:.2%}")
        print(f"   Bob & Charlie: {bob_charlie_compat:.2%}")
        print()

        # Form friendships based on compatibility
        print("🤝 Forming friendships...")
        if alice_bob_compat > 0.4:
            alice.add_friend(bob)
            print(f"   ✅ Alice and Bob are now friends!")

        if alice_charlie_compat > 0.4:
            alice.add_friend(charlie)
            print(f"   ✅ Alice and Charlie are now friends!")

        if bob_charlie_compat > 0.4:
            bob.add_friend(charlie)
            print(f"   ✅ Bob and Charlie are now friends!")
        print()

        # Generate some messages
        print("💬 Generating messages...")
        contexts = [
            "Welcome to our social network!",
            "What do you think about AI safety?",
            "Let's discuss creative applications of machine learning"
        ]

        for i, agent in enumerate(agents):
            print(f"   {agent.name}: {agent.generate_message(contexts[i])}")
        print()

        # Record some interactions
        print("📊 Recording interactions...")
        alice.record_interaction(bob.agent_id, "Hi Bob! Nice to meet you!", "greeting")
        bob.record_interaction(alice.agent_id, "Hello Alice! Great to connect.", "greeting")
        alice.record_interaction(charlie.agent_id, "I love your creative work!", "compliment")
        print()

        # Join communities
        print("🏘️ Joining communities...")
        alice.join_community("AI_Researchers")
        bob.join_community("Data_Scientists")
        charlie.join_community("Creative_AI")

        for agent in agents:
            communities = ", ".join(agent.communities)
            print(f"   {agent.name}: {communities}")
        print()

        # Show statistics
        print("📈 Agent Statistics:")
        for agent in agents:
            stats = agent.get_stats()
            print(f"   🎭 {agent.name}:")
            print(f"      Friends: {stats['total_friends']}")
            print(f"      Communities: {stats['total_communities']}")
            print(f"      Interactions: {stats['total_interactions']}")
            print(f"      Most common interaction: {stats['most_common_interaction']}")
        print()

        print("✅ Demo completed successfully!")
        print("🚀 Ready to scale to millions of agents!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())