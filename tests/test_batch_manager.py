"""
Unit tests for batch agent manager module
"""

import pytest
import time
from typing import List
from unittest.mock import Mock, patch


class TestBatchAgentManager:
    """Test batch agent creation and management functionality"""

    def test_batch_manager_creation(self, mock_env_vars):
        """Test that batch manager can be created with default settings"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()

        assert manager.max_agents == 1000000
        assert manager.batch_size == 100
        assert len(manager.agents) == 0
        assert manager.next_id == 0

    def test_batch_manager_custom_settings(self, mock_env_vars):
        """Test batch manager creation with custom settings"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager(
            max_agents=5000,
            batch_size=50
        )

        assert manager.max_agents == 5000
        assert manager.batch_size == 50

    def test_create_single_agent(self, mock_env_vars):
        """Test creating a single agent"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agent = manager.create_agent(
            name="Test Agent",
            personality="friendly",
            interests=["AI", "testing"]
        )

        assert agent is not None
        assert agent.name == "Test Agent"
        assert agent.personality == "friendly"
        assert "AI" in agent.interests
        assert "testing" in agent.interests
        assert agent.agent_id == "agent_0"
        assert len(manager.agents) == 1

    def test_create_batch_agents_default_params(self, mock_env_vars):
        """Test creating a batch of agents with default parameters"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(count=10)

        assert len(agents) == 10
        assert len(manager.agents) == 10

        # Check agent IDs are sequential
        for i, agent in enumerate(agents):
            assert agent.agent_id == f"agent_{i}"
            assert agent.name.startswith("Agent_")
            assert agent.personality in ["friendly", "analytical", "creative", "formal", "casual"]
            assert len(agent.interests) > 0

    def test_create_batch_agents_custom_params(self, mock_env_vars):
        """Test creating a batch of agents with custom parameters"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(
            count=5,
            name_prefix="CustomAgent",
            personalities=["analytical", "creative"],
            interests_list=[["AI", "ML"], ["data science", "research"]]
        )

        assert len(agents) == 5

        # Check custom parameters
        for agent in agents:
            assert agent.name.startswith("CustomAgent_")
            assert agent.personality in ["analytical", "creative"]
            assert len(agent.interests) > 0

    def test_create_batch_agents_exceeds_max(self, mock_env_vars):
        """Test error when trying to create more agents than max allowed"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager(max_agents=5)

        # Create 3 agents first
        manager.create_batch_agents(count=3)

        # Try to create 3 more (would exceed max)
        with pytest.raises(ValueError, match="Cannot create.*would exceed maximum"):
            manager.create_batch_agents(count=3)

    def test_get_agent_by_id(self, mock_env_vars):
        """Test retrieving agents by ID"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(count=3)

        # Test valid IDs
        for i, agent in enumerate(agents):
            retrieved = manager.get_agent_by_id(f"agent_{i}")
            assert retrieved is agent

        # Test invalid ID
        assert manager.get_agent_by_id("invalid_id") is None

    def test_get_agents_by_personality(self, mock_env_vars):
        """Test filtering agents by personality"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()

        # Create agents with specific personalities to ensure we have test data
        manager.create_agent("FriendlyAgent1", personality="friendly")
        manager.create_agent("FriendlyAgent2", personality="friendly")
        manager.create_agent("AnalyticalAgent1", personality="analytical")
        manager.create_agent("CreativeAgent1", personality="creative")

        friendly_agents = manager.get_agents_by_personality("friendly")
        analytical_agents = manager.get_agents_by_personality("analytical")

        assert len(friendly_agents) == 2
        assert len(analytical_agents) == 1

        for agent in friendly_agents:
            assert agent.personality == "friendly"

        for agent in analytical_agents:
            assert agent.personality == "analytical"

    def test_get_agents_by_interest(self, mock_env_vars):
        """Test filtering agents by interests"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        manager.create_batch_agents(count=10)

        # Test with more general interest terms that are likely to exist
        ai_agents = manager.get_agents_by_interest("AI")
        learning_agents = manager.get_agents_by_interest("learning")
        research_agents = manager.get_agents_by_interest("research")

        # At least one of these should have matches
        total_found = len(ai_agents) + len(learning_agents) + len(research_agents)
        assert total_found > 0, "Should find agents with at least one of the test interests"

        # Test specific interest filters that returned results
        for agent_list in [ai_agents, learning_agents, research_agents]:
            for agent in agent_list:
                # Verify that the agent actually has matching interests
                has_matching_interest = any(
                    interest_term.lower() in agent_interest.lower()
                    for agent_interest in agent.interests
                    for interest_term in ["AI", "learning", "research"]
                )
                assert has_matching_interest

    def test_batch_create_friendships(self, mock_env_vars):
        """Test creating friendships between batches of agents"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(count=10)

        # Create random friendships
        friendships_created = manager.create_random_friendships(max_friends_per_agent=3)

        assert friendships_created > 0

        # Check that some agents have friends
        agents_with_friends = [agent for agent in agents if len(agent.friends) > 0]
        assert len(agents_with_friends) > 0

    def test_batch_create_communities(self, mock_env_vars):
        """Test creating communities and assigning agents"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(count=20)

        # Create communities
        communities = manager.create_communities(
            community_names=["AI_Researchers", "Data_Scientists", "Developers"],
            max_members_per_community=8
        )

        assert len(communities) == 3

        # Check that agents are assigned to communities
        total_assignments = sum(len(agent.communities) for agent in agents)
        assert total_assignments > 0

    def test_get_manager_statistics(self, mock_env_vars):
        """Test getting statistics about the agent population"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        manager.create_batch_agents(count=50)

        stats = manager.get_statistics()

        assert stats['total_agents'] == 50
        assert 'personalities' in stats
        assert 'common_interests' in stats
        assert 'average_friends' in stats
        assert 'agents_in_communities' in stats
        assert len(stats['personalities']) > 0

    def test_batch_agent_interaction(self, mock_env_vars):
        """Test running batch interactions between agents"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(count=5)

        # Create some friendships first
        manager.create_random_friendships(max_friends_per_agent=2)

        # Run batch interactions
        interactions = manager.run_batch_interactions(
            context="Discuss AI research",
            max_interactions=10
        )

        assert len(interactions) > 0

        # Check that interactions were recorded
        for interaction in interactions:
            assert 'agent_id' in interaction
            assert 'message' in interaction
            assert len(interaction['message']) > 0

    def test_memory_usage_monitoring(self, mock_env_vars):
        """Test memory usage monitoring for large agent populations"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()

        # Create a large batch
        agents = manager.create_batch_agents(count=1000)

        memory_info = manager.get_memory_usage()

        assert 'total_agents' in memory_info
        assert 'estimated_memory_mb' in memory_info
        assert 'memory_per_agent_kb' in memory_info
        assert memory_info['total_agents'] == 1000
        assert memory_info['estimated_memory_mb'] > 0

    def test_export_agent_data(self, mock_env_vars):
        """Test exporting agent data to various formats"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agents = manager.create_batch_agents(count=5)

        # Test JSON export
        json_data = manager.export_agents(format='json')
        assert isinstance(json_data, str)
        assert len(json_data) > 0

        # Test dict export
        dict_data = manager.export_agents(format='dict')
        assert isinstance(dict_data, list)
        assert len(dict_data) == 5
        assert 'agent_id' in dict_data[0]

    def test_clear_all_agents(self, mock_env_vars):
        """Test clearing all agents from the manager"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        manager.create_batch_agents(count=10)

        assert len(manager.agents) == 10

        manager.clear_all_agents()

        assert len(manager.agents) == 0
        assert manager.next_id == 0

    def test_batch_performance_metrics(self, mock_env_vars):
        """Test performance metrics for batch operations"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()

        # Measure batch creation time
        start_time = time.time()
        agents = manager.create_batch_agents(count=100)
        creation_time = time.time() - start_time

        metrics = manager.get_performance_metrics()

        assert 'agents_created' in metrics
        assert 'creation_time_seconds' in metrics
        assert 'agents_per_second' in metrics
        assert metrics['agents_created'] == 100
        assert abs(metrics['creation_time_seconds'] - creation_time) < 0.1
        assert metrics['agents_per_second'] > 0

    def test_create_agent_when_at_max_limit(self, mock_env_vars):
        """Test creating agent when already at maximum limit"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager(max_agents=2)

        # Create agents to reach max limit
        manager.create_batch_agents(count=2)

        # Try to create one more
        with pytest.raises(ValueError, match="Cannot create agent: maximum limit"):
            manager.create_agent(name="Extra Agent")

    def test_create_agent_with_all_parameters(self, mock_env_vars):
        """Test creating agent with all custom parameters"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agent = manager.create_agent(
            name="Custom Agent",
            personality="creative",
            interests=["art", "music", "design"],
            bio="Custom biography for testing"
        )

        assert agent.name == "Custom Agent"
        assert agent.personality == "creative"
        assert agent.interests == ["art", "music", "design"]
        assert agent.bio == "Custom biography for testing"

    def test_create_communities_with_no_agents(self, mock_env_vars):
        """Test creating communities when no agents exist"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        communities = manager.create_communities(["Empty_Community"])

        assert communities == {"Empty_Community": []}

    def test_batch_interactions_with_no_agents(self, mock_env_vars):
        """Test batch interactions when no agents exist"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        interactions = manager.run_batch_interactions(max_interactions=10)

        assert interactions == []

    def test_memory_usage_with_no_agents(self, mock_env_vars):
        """Test memory usage when no agents exist"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        memory_info = manager.get_memory_usage()

        assert memory_info['total_agents'] == 0
        assert memory_info['memory_per_agent_kb'] == 0

    def test_performance_metrics_with_no_creation_time(self, mock_env_vars):
        """Test performance metrics when no creation time recorded"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        # Don't create any agents, just check metrics
        metrics = manager.get_performance_metrics()

        assert metrics['agents_created'] == 0
        assert metrics['creation_time_seconds'] == 0
        assert metrics['agents_per_second'] == 0

    def test_export_unsupported_format(self, mock_env_vars):
        """Test export with unsupported format"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        manager.create_batch_agents(count=2)

        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_agents(format='xml')

    def test_create_agent_with_none_name(self, mock_env_vars):
        """Test creating agent with name=None to test auto-generation"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agent = manager.create_agent(
            name=None,  # This should trigger auto-generation
            personality="friendly",
            interests=["test"]
        )

        assert agent.name.startswith("Agent_")
        assert agent.agent_id in manager.agents

    def test_create_agent_with_none_personality(self, mock_env_vars):
        """Test creating agent with personality=None to test default selection"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agent = manager.create_agent(
            name="Test Agent",
            personality=None,  # This should trigger random selection
            interests=["test"]
        )

        assert agent.personality in manager.DEFAULT_PERSONALITIES

    def test_create_agent_with_none_interests(self, mock_env_vars):
        """Test creating agent with interests=None to test default selection"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agent = manager.create_agent(
            name="Test Agent",
            personality="friendly",
            interests=None  # This should trigger random selection
        )

        assert len(agent.interests) > 0
        assert isinstance(agent.interests, list)

    def test_create_agent_with_none_bio(self, mock_env_vars):
        """Test creating agent with bio=None to test auto-generation"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        agent = manager.create_agent(
            name="Test Agent",
            personality="creative",
            interests=["art", "music"],
            bio=None  # This should trigger auto-generation
        )

        assert "creative" in agent.bio
        assert "art" in agent.bio or "music" in agent.bio

    def test_create_friendships_with_single_agent(self, mock_env_vars):
        """Test creating friendships when there's only one agent"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        manager.create_batch_agents(count=1)

        friendships_created = manager.create_random_friendships()

        assert friendships_created == 0

    def test_statistics_with_empty_manager(self, mock_env_vars):
        """Test getting statistics from empty manager"""
        from src.agents.batch_manager import BatchAgentManager

        manager = BatchAgentManager()
        stats = manager.get_statistics()

        assert stats['total_agents'] == 0
        assert stats['personalities'] == {}
        assert stats['common_interests'] == {}
        assert stats['average_friends'] == 0
        assert stats['agents_in_communities'] == 0

    @patch('src.agents.social_agent.SocialAgent.generate_message')
    def test_batch_interactions_with_openai_error(self, mock_generate, mock_env_vars):
        """Test batch interactions when OpenAI API fails"""
        from src.agents.batch_manager import BatchAgentManager

        # Mock OpenAI API failure
        mock_generate.side_effect = Exception("API Error")

        manager = BatchAgentManager()
        manager.create_batch_agents(count=2)

        interactions = manager.run_batch_interactions(max_interactions=5)

        # Should have 2 interactions (one per agent), not 5
        assert len(interactions) == 2
        assert all('error' in interaction for interaction in interactions)
        assert all("Error generating message" in interaction['message'] for interaction in interactions)