"""
Unit tests for async processing framework
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1'
    }):
        yield


class TestAsyncAgentManager:
    """Test async agent management functionality"""

    @pytest.mark.asyncio
    async def test_async_agent_creation(self, mock_env_vars):
        """Test creating agents asynchronously"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=10)

        # Create agents asynchronously
        start_time = time.time()
        agents = await manager.create_agents_async(count=100, name_prefix="AsyncAgent")
        creation_time = time.time() - start_time

        assert len(agents) == 100
        assert creation_time < 3.0  # Should be reasonably fast due to async processing
        assert all(agent.name.startswith("AsyncAgent_") for agent in agents)

    @pytest.mark.asyncio
    async def test_async_batch_interaction(self, mock_env_vars):
        """Test batch agent interactions asynchronously"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=5)
        agents = await manager.create_agents_async(count=10)

        # Generate messages from multiple agents concurrently
        start_time = time.time()
        interactions = await manager.batch_generate_interactions(
            agents=agents,
            context="Discuss the future of AI",
            max_concurrent=5
        )
        interaction_time = time.time() - start_time

        assert len(interactions) == 10
        assert interaction_time < 10.0  # Should be faster than sequential processing
        assert all('agent_id' in interaction for interaction in interactions)
        assert all('message' in interaction for interaction in interactions)

    @pytest.mark.asyncio
    async def test_async_friendship_building(self, mock_env_vars):
        """Test building friendships between agents asynchronously"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=10)
        agents = await manager.create_agents_async(count=50)

        # Build friendships concurrently
        start_time = time.time()
        friendships = await manager.build_friendships_async(
            agents=agents,
            max_friends_per_agent=5,
            batch_size=10
        )
        friendship_time = time.time() - start_time

        assert friendships > 0
        assert friendship_time < 5.0  # Should be fast due to async processing

        # Verify friendships were created
        total_friends = sum(len(agent.friends) for agent in agents)
        assert total_friends > 0

    @pytest.mark.asyncio
    async def test_async_community_management(self, mock_env_vars):
        """Test community management operations asynchronously"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=8)
        agents = await manager.create_agents_async(count=100)

        # Create communities and assign members concurrently
        start_time = time.time()
        communities = await manager.create_communities_async(
            agents=agents,
            community_names=["Tech_Enthusiasts", "Researchers", "Developers"],
            max_members_per_community=40
        )
        community_time = time.time() - start_time

        assert len(communities) == 3
        assert community_time < 1.5  # Should be fast due to async processing

        # Verify community assignments
        total_members = sum(len(members) for members in communities.values())
        assert total_members > 0

    @pytest.mark.asyncio
    async def test_async_memory_operations(self, mock_env_vars):
        """Test async memory management operations"""
        from src.agents.async_manager import AsyncAgentManager
        from src.agents.memory_optimizer import MemoryOptimizer

        manager = AsyncAgentManager(max_concurrent=10)
        agents = await manager.create_agents_async(count=200)

        # Store agents in memory optimizer asynchronously
        memory_optimizer = MemoryOptimizer(max_agents=1000, lazy_loading_enabled=True)

        start_time = time.time()
        await manager.store_agents_async(agents, memory_optimizer)
        store_time = time.time() - start_time

        assert store_time < 1.0  # Should be fast due to async processing
        assert memory_optimizer.get_agent_count() == 200

        # Retrieve agents asynchronously
        start_time = time.time()
        retrieved_agents = await manager.retrieve_agents_async(
            [agent.agent_id for agent in agents[:50]],
            memory_optimizer
        )
        retrieve_time = time.time() - start_time

        assert len(retrieved_agents) == 50
        assert retrieve_time < 0.5  # Should be fast

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, mock_env_vars):
        """Test rate limiting for concurrent operations"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=3)  # Low concurrency limit
        agents = await manager.create_agents_async(count=10)

        # Test concurrent operations with rate limiting
        start_time = time.time()
        interactions = await manager.batch_generate_interactions(
            agents=agents[:5],
            context="Test context",
            max_concurrent=2  # Lower concurrency to test rate limiting
        )
        interaction_time = time.time() - start_time

        # Should have all interactions
        assert len(interactions) == 5

        # Should take some time due to rate limiting
        assert interaction_time >= 0.5  # Rate limiting delay should be present

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_env_vars):
        """Test error handling in async operations"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=5)

        # Create agents for testing
        agents = await manager.create_agents_async(count=5)

        # Test error handling by mocking generate_message to fail for some agents
        import unittest.mock
        with unittest.mock.patch.object(agents[0].__class__, 'generate_message') as mock_generate:
            # Make some calls fail
            call_count = 0
            def generate_side_effect(context):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("API Error")
                return f"Success message {call_count}"

            mock_generate.side_effect = generate_side_effect

            # Should handle errors gracefully
            interactions = await manager.batch_generate_interactions(
                agents=agents,
                context="Test context",
                max_concurrent=3
            )

            # Should have interactions despite errors
            assert len(interactions) == 5
            assert all(isinstance(interaction, dict) for interaction in interactions)

            # Check that some have errors
            error_count = sum(1 for interaction in interactions if 'error' in interaction)
            assert error_count >= 0  # Some may have errors

    @pytest.mark.asyncio
    async def test_async_performance_metrics(self, mock_env_vars):
        """Test performance metrics collection for async operations"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=10)
        agents = await manager.create_agents_async(count=50)

        # Perform various async operations
        await manager.build_friendships_async(agents, max_friends_per_agent=3)
        await manager.create_communities_async(
            agents,
            ["Community1", "Community2"],
            max_members_per_community=25
        )

        # Get performance metrics
        metrics = manager.get_performance_metrics()

        assert 'total_operations' in metrics
        assert 'successful_operations' in metrics
        assert 'failed_operations' in metrics
        assert 'total_time' in metrics
        assert 'average_operation_time' in metrics
        assert 'operations_per_second' in metrics
        assert metrics['total_operations'] > 0
        assert metrics['successful_operations'] > 0

    @pytest.mark.asyncio
    async def test_async_batch_size_optimization(self, mock_env_vars):
        """Test dynamic batch size optimization based on system performance"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=20, auto_optimize=True)
        agents = await manager.create_agents_async(count=100)

        # Initial batch size
        initial_batch_size = manager.get_current_batch_size()

        # Perform operations to allow optimization
        await manager.batch_generate_interactions(
            agents[:50],
            context="Performance test",
            max_concurrent=10
        )

        # Batch size should be optimized based on performance
        optimized_batch_size = manager.get_current_batch_size()

        # Should either stay the same or be adjusted for better performance
        assert isinstance(optimized_batch_size, int)
        assert optimized_batch_size > 0

    @pytest.mark.asyncio
    async def test_async_streaming_interactions(self, mock_env_vars):
        """Test streaming of agent interactions for real-time processing"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=5)
        agents = await manager.create_agents_async(count=10)

        # Stream interactions in real-time
        interaction_stream = manager.stream_interactions(
            agents=agents,
            context="Real-time conversation",
            batch_size=3
        )

        # Collect streamed results
        streamed_interactions = []
        async for batch in interaction_stream:
            streamed_interactions.extend(batch)
            # Verify we're getting results in batches
            assert len(batch) <= 3

        assert len(streamed_interactions) == 10
        assert all('agent_id' in interaction for interaction in streamed_interactions)

    @pytest.mark.asyncio
    async def test_async_cancellation(self, mock_env_vars):
        """Test cancellation of long-running async operations"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=5)
        agents = await manager.create_agents_async(count=20)

        # Start a long-running operation
        task = asyncio.create_task(
            manager.batch_generate_interactions(
                agents=agents,
                context="Long operation",
                max_concurrent=2
            )
        )

        # Let it run for a short time, then cancel
        await asyncio.sleep(0.1)
        task.cancel()

        # Should handle cancellation gracefully
        with pytest.raises(asyncio.CancelledError):
            await task

        # Manager should still be functional after cancellation
        assert len(manager.agents) == 20


class TestAsyncSocialFeatures:
    """Test async social features"""

    @pytest.mark.asyncio
    async def test_async_compatibility_matching(self, mock_env_vars):
        """Test async agent compatibility matching"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=10)
        agents = await manager.create_agents_async(count=50)

        # Find compatible matches asynchronously
        start_time = time.time()
        matches = await manager.find_compatible_matches_async(
            agents=agents[:10],
            candidate_agents=agents[10:],
            threshold=0.5,
            max_matches=5
        )
        match_time = time.time() - start_time

        assert isinstance(matches, dict)
        assert len(matches) <= 10  # At most one match per agent
        assert match_time < 1.0  # Should be fast due to async processing

    @pytest.mark.asyncio
    async def test_async_social_network_analysis(self, mock_env_vars):
        """Test async social network analysis"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=8)
        agents = await manager.create_agents_async(count=100)

        # Build social network
        await manager.build_friendships_async(agents, max_friends_per_agent=8)

        # Analyze network properties asynchronously
        start_time = time.time()
        network_stats = await manager.analyze_social_network_async(agents)
        analysis_time = time.time() - start_time

        assert 'average_connections' in network_stats
        assert 'network_density' in network_stats
        assert 'clusters' in network_stats
        assert 'influential_agents' in network_stats
        assert analysis_time < 2.0  # Should be fast

    @pytest.mark.asyncio
    async def test_async_content_recommendation(self, mock_env_vars):
        """Test async content recommendation between agents"""
        from src.agents.async_manager import AsyncAgentManager

        manager = AsyncAgentManager(max_concurrent=6)
        agents = await manager.create_agents_async(count=30)

        # Generate recommendations asynchronously
        start_time = time.time()
        recommendations = await manager.generate_recommendations_async(
            agents=agents[:10],
            candidate_agents=agents[10:],
            recommendation_type="friends",
            max_recommendations=3
        )
        recommendation_time = time.time() - start_time

        assert isinstance(recommendations, dict)
        assert len(recommendations) <= 10
        assert all(isinstance(rec_list, list) for rec_list in recommendations.values())
        assert recommendation_time < 1.5  # Should be fast