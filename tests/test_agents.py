"""
Unit tests for agent classes
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio


class TestBaseAgent:
    """Test base agent functionality"""

    def test_base_agent_creation(self, mock_env_vars):
        """Test that base agent can be created"""
        from src.agents.base import BaseAgent

        agent = BaseAgent(
            agent_id="test_agent_1",
            name="Test Agent",
            role="tester"
        )

        assert agent.agent_id == "test_agent_1"
        assert agent.name == "Test Agent"
        assert agent.role == "tester"

    def test_base_agent_unique_ids(self, mock_env_vars):
        """Test that agents have unique IDs"""
        from src.agents.base import BaseAgent

        agent1 = BaseAgent(agent_id="agent_1")
        agent2 = BaseAgent(agent_id="agent_2")

        assert agent1.agent_id != agent2.agent_id
        assert agent1 != agent2

    def test_base_agent_string_representation(self, mock_env_vars):
        """Test string representation of base agent"""
        from src.agents.base import BaseAgent

        agent = BaseAgent(
            agent_id="test_agent_1",
            name="Test Agent"
        )

        agent_str = str(agent)
        assert "test_agent_1" in agent_str
        assert "Test Agent" in agent_str

    def test_base_agent_properties(self, mock_env_vars):
        """Test agent properties"""
        from src.agents.base import BaseAgent

        agent = BaseAgent(
            agent_id="test_agent",
            name="Test",
            role="tester",
            description="A test agent"
        )

        assert agent.agent_id == "test_agent"
        assert agent.name == "Test"
        assert agent.role == "tester"
        assert agent.description == "A test agent"

    def test_base_agent_equality(self, mock_env_vars):
        """Test agent equality based on ID"""
        from src.agents.base import BaseAgent

        agent1 = BaseAgent(agent_id="same_id")
        agent2 = BaseAgent(agent_id="same_id")
        agent3 = BaseAgent(agent_id="different_id")

        assert agent1 == agent2
        assert agent1 != agent3

    def test_base_agent_hash(self, mock_env_vars):
        """Test agent hash for use in sets/dicts"""
        from src.agents.base import BaseAgent

        agent = BaseAgent(agent_id="test_id")

        # Should be hashable
        agent_hash = hash(agent)
        assert isinstance(agent_hash, int)

        # Same ID should have same hash
        agent2 = BaseAgent(agent_id="test_id")
        assert hash(agent) == hash(agent2)


class TestSocialAgent:
    """Test social agent functionality"""

    def test_social_agent_creation(self, mock_env_vars):
        """Test that social agent can be created"""
        from src.agents.social_agent import SocialAgent

        agent = SocialAgent(
            agent_id="social_agent_1",
            name="Social Agent",
            personality="friendly",
            interests=["AI", "social networks"]
        )

        assert agent.agent_id == "social_agent_1"
        assert agent.name == "Social Agent"
        assert agent.personality == "friendly"
        assert "AI" in agent.interests
        assert "social networks" in agent.interests

    def test_social_agent_inherits_base(self, mock_env_vars):
        """Test that social agent inherits from base agent"""
        from src.agents.social_agent import SocialAgent
        from src.agents.base import BaseAgent

        agent = SocialAgent(agent_id="test_id")

        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, 'agent_id')
        assert hasattr(agent, 'name')

    def test_social_agent_friends_management(self, mock_env_vars):
        """Test social agent friend management"""
        from src.agents.social_agent import SocialAgent

        agent1 = SocialAgent(agent_id="agent_1")
        agent2 = SocialAgent(agent_id="agent_2")

        # Initially no friends
        assert len(agent1.friends) == 0

        # Add friend
        agent1.add_friend(agent2)
        assert len(agent1.friends) == 1
        assert agent2 in agent1.friends

        # Remove friend
        agent1.remove_friend(agent2)
        assert len(agent1.friends) == 0
        assert agent2 not in agent1.friends

    def test_social_agent_duplicate_friends(self, mock_env_vars):
        """Test that duplicate friends are not added"""
        from src.agents.social_agent import SocialAgent

        agent1 = SocialAgent(agent_id="agent_1")
        agent2 = SocialAgent(agent_id="agent_2")

        # Add same friend twice
        agent1.add_friend(agent2)
        agent1.add_friend(agent2)

        assert len(agent1.friends) == 1
        assert agent2 in agent1.friends

    def test_social_agent_communities(self, mock_env_vars):
        """Test social agent community membership"""
        from src.agents.social_agent import SocialAgent

        agent = SocialAgent(agent_id="test_agent")

        # Initially no communities
        assert len(agent.communities) == 0

        # Join community
        agent.join_community("AI_Researchers")
        assert "AI_Researchers" in agent.communities

        # Leave community
        agent.leave_community("AI_Researchers")
        assert "AI_Researchers" not in agent.communities

    def test_social_agent_interaction_history(self, mock_env_vars):
        """Test social agent interaction history tracking"""
        from src.agents.social_agent import SocialAgent

        agent1 = SocialAgent(agent_id="agent_1")
        agent2 = SocialAgent(agent_id="agent_2")

        # Record interaction
        agent1.record_interaction(agent2.agent_id, "Hello!", "greeting")

        assert len(agent1.interaction_history) == 1
        interaction = agent1.interaction_history[0]
        assert interaction["with"] == agent2.agent_id
        assert interaction["message"] == "Hello!"
        assert interaction["type"] == "greeting"

    def test_social_agent_personality_validation(self, mock_env_vars):
        """Test social agent personality validation"""
        from src.agents.social_agent import SocialAgent

        # Valid personalities
        valid_personalities = ["friendly", "analytical", "creative", "formal"]
        for personality in valid_personalities:
            agent = SocialAgent(agent_id=f"agent_{personality}", personality=personality)
            assert agent.personality == personality

        # Invalid personality should raise error
        with pytest.raises(ValueError, match="Invalid personality"):
            SocialAgent(agent_id="agent_invalid", personality="invalid_personality")

    def test_social_agent_compatibility_check(self, mock_env_vars):
        """Test social agent compatibility checking"""
        from src.agents.social_agent import SocialAgent

        agent1 = SocialAgent(
            agent_id="agent_1",
            personality="friendly",
            interests=["AI", "machine learning"]
        )
        agent2 = SocialAgent(
            agent_id="agent_2",
            personality="friendly",
            interests=["AI", "social networks"]
        )
        agent3 = SocialAgent(
            agent_id="agent_3",
            personality="analytical",
            interests=["math", "statistics"]
        )

        # Compatible agents (same personality, shared interests)
        compatibility_1_2 = agent1.check_compatibility(agent2)
        assert compatibility_1_2 > 0.5  # Should be compatible

        # Less compatible agents (different personality, no shared interests)
        compatibility_1_3 = agent1.check_compatibility(agent3)
        assert compatibility_1_3 < compatibility_1_2

    @patch('src.agents.social_agent.openai.OpenAI')
    def test_social_agent_generate_message(self, mock_openai, mock_env_vars):
        """Test social agent message generation"""
        from src.agents.social_agent import SocialAgent

        # Setup mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, nice to meet you!"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = SocialAgent(
            agent_id="test_agent",
            personality="friendly",
            interests=["AI", "social networks"]
        )

        # Generate message
        message = agent.generate_message(context="Hello! First meeting")

        assert isinstance(message, str)
        assert len(message) > 0

    def test_social_agent_stats(self, mock_env_vars):
        """Test social agent statistics"""
        from src.agents.social_agent import SocialAgent

        agent1 = SocialAgent(agent_id="agent_1")
        agent2 = SocialAgent(agent_id="agent_2")

        # Add some interactions
        agent1.record_interaction(agent2.agent_id, "Hi!", "greeting")
        agent1.record_interaction(agent2.agent_id, "How are you?", "question")

        # Add friend and community
        agent1.add_friend(agent2)
        agent1.join_community("test_community")

        stats = agent1.get_stats()

        assert stats["total_interactions"] == 2
        assert stats["total_friends"] == 1
        assert stats["total_communities"] == 1
        assert "interaction_partners" in stats
        assert agent2.agent_id in stats["interaction_partners"]