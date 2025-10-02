"""
Unit tests for memory optimization module
"""

import pytest
import sys
import gc
import time
import random
from unittest.mock import Mock, patch


class TestMemoryOptimizer:
    """Test memory optimization for large-scale agent populations"""

    def test_memory_optimizer_creation(self, mock_env_vars):
        """Test memory optimizer creation with default settings"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        assert optimizer.max_agents == 1000000
        assert optimizer.compression_enabled == True
        assert optimizer.lazy_loading_enabled == True
        assert len(optimizer.agent_cache) == 0
        assert len(optimizer.loaded_agents) == 0

    def test_memory_optimizer_custom_settings(self, mock_env_vars):
        """Test memory optimizer with custom settings"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer(
            max_agents=500000,
            compression_enabled=False,
            lazy_loading_enabled=False,
            cache_size=1000
        )

        assert optimizer.max_agents == 500000
        assert optimizer.compression_enabled == False
        assert optimizer.lazy_loading_enabled == False
        assert optimizer.cache_size == 1000

    def test_agent_compression(self, mock_env_vars):
        """Test agent data compression and decompression"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(compression_enabled=True)

        # Create test agent
        agent = SocialAgent(
            agent_id="test_001",
            name="Test Agent",
            personality="friendly",
            interests=["AI", "testing"],
            bio="Test agent for compression"
        )

        # Compress agent
        compressed_data = optimizer.compress_agent(agent)

        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        assert b'test_001' in compressed_data

        # Decompress agent
        decompressed_agent = optimizer.decompress_agent(compressed_data)

        assert decompressed_agent.agent_id == agent.agent_id
        assert decompressed_agent.name == agent.name
        assert decompressed_agent.personality == agent.personality
        assert decompressed_agent.interests == agent.interests
        assert decompressed_agent.bio == agent.bio

    def test_agent_storage_without_compression(self, mock_env_vars):
        """Test agent storage without compression"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(compression_enabled=False)

        agent = SocialAgent(
            agent_id="test_002",
            name="Test Agent 2",
            personality="analytical",
            interests=["data", "science"]
        )

        # Store agent without compression
        stored_data = optimizer.store_agent_data(agent)

        assert isinstance(stored_data, dict)
        assert stored_data['agent_id'] == agent.agent_id
        assert stored_data['name'] == agent.name

        # Retrieve agent
        retrieved_agent = optimizer.retrieve_agent_data(stored_data)

        assert retrieved_agent.agent_id == agent.agent_id
        assert retrieved_agent.name == agent.name

    def test_lazy_loading_agent(self, mock_env_vars):
        """Test lazy loading of agent data"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(lazy_loading_enabled=True, cache_size=2)

        # Create and store multiple agents
        agents = []
        for i in range(5):
            agent = SocialAgent(
                agent_id=f"lazy_test_{i}",
                name=f"Lazy Agent {i}",
                personality="friendly",
                interests=[f"interest_{i}"]
            )
            agents.append(agent)
            optimizer.store_agent(agent)

        # Test lazy loading - only recently accessed agents should be in memory
        agent_0 = optimizer.get_agent("lazy_test_0")
        agent_1 = optimizer.get_agent("lazy_test_1")

        assert len(optimizer.loaded_agents) <= 2  # Cache size limit
        assert agent_0.agent_id == "lazy_test_0"
        assert agent_1.agent_id == "lazy_test_1"

        # Access another agent should trigger cache eviction
        agent_2 = optimizer.get_agent("lazy_test_2")

        assert agent_2.agent_id == "lazy_test_2"
        assert len(optimizer.loaded_agents) <= 2

    def test_batch_agent_storage(self, mock_env_vars):
        """Test storing and retrieving large batches of agents"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Create batch of agents
        agents = []
        for i in range(100):
            agent = SocialAgent(
                agent_id=f"batch_{i:03d}",
                name=f"Batch Agent {i}",
                personality=random.choice(["friendly", "analytical", "creative"]),
                interests=[f"interest_{i % 10}"]
            )
            agents.append(agent)

        # Store all agents
        storage_time_start = time.time()
        optimizer.store_batch_agents(agents)
        storage_time = time.time() - storage_time_start

        # Verify storage
        assert optimizer.total_stored_agents == 100
        assert storage_time < 5.0  # Should be fast

        # Retrieve a sample
        retrieved_agent = optimizer.get_agent("batch_050")
        assert retrieved_agent.agent_id == "batch_050"
        assert retrieved_agent.name == "Batch Agent 50"

    def test_memory_usage_monitoring(self, mock_env_vars):
        """Test memory usage monitoring and optimization"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Get initial memory usage
        initial_memory = optimizer.get_memory_usage()
        assert initial_memory['total_agents'] == 0

        # Store many agents
        for i in range(1000):
            agent = SocialAgent(
                agent_id=f"memory_test_{i}",
                name=f"Memory Test {i}",
                personality="friendly",
                interests=[f"hobby_{i % 5}"]
            )
            optimizer.store_agent(agent)

        # Check memory usage after storing
        memory_after = optimizer.get_memory_usage()

        assert memory_after['total_agents'] == 1000
        assert memory_after['compressed_size_mb'] > 0
        assert memory_after['cache_size_mb'] >= 0
        assert memory_after['memory_saved_percent'] >= 0

        # Test memory cleanup
        optimizer.cleanup_memory()

        memory_after_cleanup = optimizer.get_memory_usage()
        assert memory_after_cleanup['total_agents'] == 1000  # Agents still stored

    def test_agent_indexing(self, mock_env_vars):
        """Test fast agent indexing and lookup"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Create agents with different attributes
        personalities = ["friendly", "analytical", "creative", "formal"]
        interests = ["AI", "data", "art", "science", "music"]

        agents = []
        for i in range(200):
            agent = SocialAgent(
                agent_id=f"index_test_{i}",
                name=f"Index Test {i}",
                personality=random.choice(personalities),
                interests=random.sample(interests, 2)
            )
            agents.append(agent)
            optimizer.store_agent(agent)

        # Build indexes
        optimizer.build_indexes()

        # Test personality lookup
        friendly_agents = optimizer.find_by_personality("friendly")
        assert len(friendly_agents) > 0
        assert all(agent.personality == "friendly" for agent in friendly_agents)

        # Test interest lookup
        ai_agents = optimizer.find_by_interest("AI")
        assert len(ai_agents) > 0
        assert all(any("AI" in interest for interest in agent.interests) for agent in ai_agents)

        # Test range lookup
        range_agents = optimizer.find_by_id_range("index_test_50", "index_test_59")
        assert len(range_agents) == 10

    def test_cache_eviction_policy(self, mock_env_vars):
        """Test cache eviction when memory limit is reached"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(cache_size=3)

        # Store and access more agents than cache size
        for i in range(10):
            agent = SocialAgent(
                agent_id=f"cache_test_{i}",
                name=f"Cache Test {i}",
                personality="friendly",
                interests=[f"interest_{i}"]
            )
            optimizer.store_agent(agent)

            # Access agent to load into cache
            optimizer.get_agent(f"cache_test_{i}")

        # Verify cache size is maintained
        assert len(optimizer.loaded_agents) <= 3

        # Verify most recently accessed agents are in cache
        # Last accessed agents should be cache_test_7, cache_test_8, cache_test_9
        recent_ids = {"cache_test_7", "cache_test_8", "cache_test_9"}
        cached_ids = set(agent.agent_id for agent in optimizer.loaded_agents.values())

        assert cached_ids.issubset(recent_ids)

    def test_data_integrity_verification(self, mock_env_vars):
        """Test data integrity verification for stored agents"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store test agent
        original_agent = SocialAgent(
            agent_id="integrity_test",
            name="Integrity Test",
            personality="analytical",
            interests=["testing", "verification"],
            bio="Agent for integrity testing"
        )
        optimizer.store_agent(original_agent)

        # Verify integrity
        is_valid = optimizer.verify_agent_integrity("integrity_test")
        assert is_valid == True

        # Test non-existent agent
        is_valid_invalid = optimizer.verify_agent_integrity("non_existent")
        assert is_valid_invalid == False

    def test_memory_compaction(self, mock_env_vars):
        """Test memory compaction and garbage collection"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store many agents
        for i in range(500):
            agent = SocialAgent(
                agent_id=f"compact_test_{i}",
                name=f"Compact Test {i}",
                personality="friendly",
                interests=[f"test_{i}"]
            )
            optimizer.store_agent(agent)

        memory_before = optimizer.get_memory_usage()

        # Remove some agents
        for i in range(100):
            optimizer.remove_agent(f"compact_test_{i}")

        # Run compaction
        optimizer.compact_memory()

        memory_after = optimizer.get_memory_usage()

        assert memory_after['total_agents'] == 400  # 500 - 100 removed
        assert memory_after['fragmentation_percent'] < memory_before['fragmentation_percent']

    def test_parallel_agent_access(self, mock_env_vars):
        """Test concurrent access to agents"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        import concurrent.futures
        import threading

        optimizer = MemoryOptimizer()

        # Store test agents
        for i in range(50):
            agent = SocialAgent(
                agent_id=f"parallel_test_{i}",
                name=f"Parallel Test {i}",
                personality="friendly",
                interests=[f"interest_{i}"]
            )
            optimizer.store_agent(agent)

        # Test concurrent access
        def access_agents(start_idx, end_idx):
            results = []
            for i in range(start_idx, end_idx):
                agent = optimizer.get_agent(f"parallel_test_{i}")
                results.append(agent.agent_id if agent else None)
            return results

        # Run concurrent access
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(0, 50, 10):
                future = executor.submit(access_agents, i, i + 10)
                futures.append(future)

            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        # Verify all agents were accessed successfully
        assert len(all_results) == 50
        assert all(result is not None for result in all_results)

    def test_memory_optimization_statistics(self, mock_env_vars):
        """Test memory optimization statistics and reporting"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store agents with different characteristics
        for i in range(100):
            agent = SocialAgent(
                agent_id=f"stats_test_{i}",
                name=f"Stats Test {i}",
                personality=random.choice(["friendly", "analytical", "creative", "formal"]),
                interests=random.sample(["AI", "data", "art", "science"], 2)
            )
            optimizer.store_agent(agent)

        # Get statistics
        stats = optimizer.get_optimization_statistics()

        assert 'total_agents' in stats
        assert 'compression_ratio' in stats
        assert 'cache_hit_ratio' in stats
        assert 'memory_efficiency' in stats
        assert 'storage_overhead' in stats

        assert stats['total_agents'] == 100
        assert 0 <= stats['compression_ratio'] <= 1
        assert 0 <= stats['cache_hit_ratio'] <= 1

    def test_compression_fallback_to_pickle(self, mock_env_vars):
        """Test compression fallback when msgpack is not available"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        import unittest.mock

        optimizer = MemoryOptimizer(compression_enabled=True)

        # Create test agent
        agent = SocialAgent(
            agent_id="fallback_test",
            name="Fallback Test",
            personality="friendly",
            interests=["testing", "compression"]
        )

        # Mock msgpack to be unavailable
        with unittest.mock.patch('src.agents.memory_optimizer.HAS_MSGPACK', False):
            compressed_data = optimizer.compress_agent(agent)
            # Should use pickle when msgpack is not available
            assert isinstance(compressed_data, bytes)
            assert len(compressed_data) > 0

    def test_decompression_msgpack_exception_fallback(self, mock_env_vars):
        """Test decompression fallback when msgpack fails"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        import unittest.mock
        import pickle

        # Create optimizer with compression disabled to use pickle
        optimizer = MemoryOptimizer(compression_enabled=False)

        # Create test agent
        agent = SocialAgent(
            agent_id="exception_test",
            name="Exception Test",
            personality="analytical",
            interests=["error", "testing"]
        )

        # Compress with pickle (not msgpack)
        compressed_data = optimizer.compress_agent(agent)

        # Now enable compression and mock msgpack to test fallback
        optimizer.compression_enabled = True

        # Mock msgpack to be available but fail during unpacking
        with unittest.mock.patch('src.agents.memory_optimizer.HAS_MSGPACK', True):
            with unittest.mock.patch('msgpack.unpackb', side_effect=Exception("Mocked msgpack failure")):
                # Should fall back to pickle
                decompressed_agent = optimizer.decompress_agent(compressed_data)
                assert decompressed_agent.agent_id == agent.agent_id
                assert decompressed_agent.name == agent.name

    def test_decompression_without_msgpack(self, mock_env_vars):
        """Test decompression when msgpack is not available"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        import unittest.mock

        optimizer = MemoryOptimizer(compression_enabled=False)

        # Create test agent
        agent = SocialAgent(
            agent_id="no_msgpack_test",
            name="No Msgpack Test",
            personality="creative",
            interests=["art", "design"]
        )

        # Store without compression
        compressed_data = optimizer.compress_agent(agent)

        # Mock msgpack to be unavailable
        with unittest.mock.patch('src.agents.memory_optimizer.HAS_MSGPACK', False):
            decompressed_agent = optimizer.decompress_agent(compressed_data)
            assert decompressed_agent.agent_id == agent.agent_id
            assert decompressed_agent.name == agent.name

    def test_storage_at_max_limit(self, mock_env_vars):
        """Test behavior when reaching maximum agent limit"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(max_agents=2)

        # Store agents up to limit
        agent1 = SocialAgent("limit_test_1", "Test 1", "friendly", ["test"])
        agent2 = SocialAgent("limit_test_2", "Test 2", "analytical", ["test"])

        optimizer.store_agent(agent1)
        optimizer.store_agent(agent2)

        # Should raise error when trying to store beyond limit
        agent3 = SocialAgent("limit_test_3", "Test 3", "creative", ["test"])
        with pytest.raises(ValueError, match="Maximum agent limit"):
            optimizer.store_agent(agent3)

    def test_get_nonexistent_agent(self, mock_env_vars):
        """Test getting an agent that doesn't exist"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Should return None for non-existent agent
        result = optimizer.get_agent("nonexistent_agent")
        assert result is None

    def test_find_by_personality_with_no_matches(self, mock_env_vars):
        """Test finding agents by personality when no matches exist"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()
        optimizer.build_indexes()

        # Should return empty list when no agents match
        result = optimizer.find_by_personality("nonexistent_personality")
        assert result == []

    def test_find_by_interest_with_no_matches(self, mock_env_vars):
        """Test finding agents by interest when no matches exist"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()
        optimizer.build_indexes()

        # Should return empty list when no agents match
        result = optimizer.find_by_interest("nonexistent_interest")
        assert result == []

    def test_find_by_id_range_with_no_matches(self, mock_env_vars):
        """Test finding agents by ID range when no matches exist"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()
        optimizer.build_indexes()

        # Should return empty list when no agents match
        result = optimizer.find_by_id_range("agent_999", "agent_1000")
        assert result == []

    def test_decompression_with_corrupted_data(self, mock_env_vars):
        """Test decompression with corrupted data"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Try to decompress corrupted data
        corrupted_data = b"corrupted_data_that_cannot_be_decompressed"

        with pytest.raises(ValueError, match="Failed to decompress agent data"):
            optimizer.decompress_agent(corrupted_data)

    def test_remove_nonexistent_agent(self, mock_env_vars):
        """Test removing an agent that doesn't exist"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Should return False when trying to remove non-existent agent
        result = optimizer.remove_agent("nonexistent_agent")
        assert result is False

    def test_cache_behavior_without_lazy_loading(self, mock_env_vars):
        """Test cache behavior when lazy loading is disabled"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(lazy_loading_enabled=False)

        # Create and store agent
        agent = SocialAgent("cache_test", "Cache Test", "friendly", ["test"])
        optimizer.store_agent(agent)

        # Retrieve agent multiple times
        agent1 = optimizer.get_agent("cache_test")
        agent2 = optimizer.get_agent("cache_test")

        # Should return the same agent
        assert agent1.agent_id == agent2.agent_id
        assert agent1.name == agent2.name

    def test_build_indexes_with_corrupted_data(self, mock_env_vars):
        """Test building indexes when some stored data is corrupted"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store a valid agent
        agent = SocialAgent("valid_agent", "Valid Agent", "friendly", ["test"])
        optimizer.store_agent(agent)

        # Add corrupted data directly to storage
        optimizer.agent_storage["corrupted_agent"] = b"corrupted_data"

        # Build indexes should skip corrupted data
        optimizer.build_indexes()

        # Should still have the valid agent indexed
        assert "valid_agent" in optimizer.name_index.values()
        assert "corrupted_agent" not in optimizer.name_index.values()

    def test_compaction_with_corrupted_data(self, mock_env_vars):
        """Test memory compaction with corrupted data"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store a valid agent
        agent = SocialAgent("compact_valid", "Valid Agent", "friendly", ["test"])
        optimizer.store_agent(agent)

        # Add corrupted data directly to storage
        optimizer.agent_storage["compact_corrupted"] = b"corrupted_data"

        # Verify corrupted data exists
        assert not optimizer.verify_agent_integrity("compact_corrupted")

        # Run compaction should remove corrupted data
        optimizer.compact_memory()

        # Should only have valid agent remaining
        assert len(optimizer.agent_storage) == 1
        assert "compact_valid" in optimizer.agent_storage
        assert "compact_corrupted" not in optimizer.agent_storage

    def test_cache_update_existing_agent(self, mock_env_vars):
        """Test updating cache when agent already exists"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(lazy_loading_enabled=True, cache_size=5)

        # Store and access agent to add to cache
        agent = SocialAgent("cache_update_test", "Cache Update Test", "friendly", ["test"])
        optimizer.store_agent(agent)
        cached_agent1 = optimizer.get_agent("cache_update_test")

        # Access same agent again (should trigger cache update logic)
        cached_agent2 = optimizer.get_agent("cache_update_test")

        # Should get the same agent
        assert cached_agent1.agent_id == cached_agent2.agent_id
        assert "cache_update_test" in optimizer.agent_cache

    def test_remove_agent_with_loaded_agents(self, mock_env_vars):
        """Test removing agent that exists in loaded_agents"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer(lazy_loading_enabled=True)

        # Store agent and manually add to loaded_agents
        agent = SocialAgent("loaded_test", "Loaded Test", "friendly", ["test"])
        optimizer.store_agent(agent)
        optimizer.loaded_agents["loaded_test"] = agent

        # Remove agent (should also remove from loaded_agents)
        result = optimizer.remove_agent("loaded_test")

        assert result is True
        assert "loaded_test" not in optimizer.agent_storage
        assert "loaded_test" not in optimizer.loaded_agents

    def test_remove_agent_cleanup_indexes(self, mock_env_vars):
        """Test removing agent cleans up empty personality indexes"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store agent with unique personality (use valid personality)
        agent = SocialAgent("unique_test", "Unique Test", "helpful", ["test"])
        optimizer.store_agent(agent)
        optimizer.build_indexes()

        # Manually add a unique personality to test cleanup
        optimizer.personality_index["unique_personality"] = {"unique_test"}

        # Verify index exists
        assert "unique_personality" in optimizer.personality_index
        assert len(optimizer.personality_index["unique_personality"]) == 1

        # Remove agent (should clean up empty index)
        result = optimizer.remove_agent("unique_test")

        assert result is True
        assert "unique_personality" not in optimizer.personality_index

    def test_remove_agent_cleanup_name_index(self, mock_env_vars):
        """Test removing agent cleans up name index"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent

        optimizer = MemoryOptimizer()

        # Store agent and build indexes
        agent = SocialAgent("name_test", "Name Test", "friendly", ["test"])
        optimizer.store_agent(agent)
        optimizer.build_indexes()

        # Verify name index exists
        assert "Name Test" in optimizer.name_index
        assert optimizer.name_index["Name Test"] == "name_test"

        # Remove agent (should clean up name index)
        result = optimizer.remove_agent("name_test")

        assert result is True
        # The name index should be cleaned up after removal
        # Let's check that the agent is actually removed from storage first
        assert "name_test" not in optimizer.agent_storage
        # The name index cleanup happens during removal, so let's verify it's gone
        assert "Name Test" not in optimizer.name_index

    def test_fragmentation_calculation_default_size(self, mock_env_vars):
        """Test fragmentation calculation with default size estimation"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Set up compression stats to zero to trigger default estimation
        optimizer.compression_stats['compressed_size'] = 0
        optimizer.agent_storage['test'] = b'data'  # Add some data

        fragmentation = optimizer._calculate_fragmentation()
        assert isinstance(fragmentation, float)
        assert fragmentation >= 0.0

    def test_fragmentation_calculation_boundary_condition(self, mock_env_vars):
        """Test fragmentation calculation boundary condition"""
        from src.agents.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Set up to trigger the boundary condition where total_estimated_size <= optimal_size
        # Need to set up conditions where overhead is minimal
        optimizer.agent_storage['test'] = b'x' * 1000  # Larger data
        optimizer.compression_stats['compressed_size'] = 2000  # Set compressed size
        optimizer.total_stored_agents = 1

        fragmentation = optimizer._calculate_fragmentation()
        # In this case, fragmentation should be 0 because total_estimated_size <= optimal_size
        assert fragmentation >= 0.0

    def test_compaction_recompression_failure(self, mock_env_vars):
        """Test memory compaction when recompression fails"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        import unittest.mock

        optimizer = MemoryOptimizer(compression_enabled=True)

        # Store agent
        agent = SocialAgent("recompress_fail", "Recompress Fail", "friendly", ["test"])
        optimizer.store_agent(agent)

        # Mock compress_agent to raise exception during recompression
        original_compress = optimizer.compress_agent
        call_count = 0

        def failing_compress(agent):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_compress(agent)  # First call succeeds
            else:
                raise Exception("Recompression failed")

        with unittest.mock.patch.object(optimizer, 'compress_agent', side_effect=failing_compress):
            # Run compaction - should keep original data when recompression fails
            optimizer.compact_memory()

        # Agent should still exist (kept with original compression)
        assert "recompress_fail" in optimizer.agent_storage
        retrieved = optimizer.get_agent("recompress_fail")
        assert retrieved.agent_id == "recompress_fail"

    def test_cache_deletion_when_updating_existing_agent(self):
        """Test cache deletion when updating existing agent in _update_cache"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        optimizer = MemoryOptimizer(max_agents=3, lazy_loading_enabled=True)

        # Store agent to populate cache
        agent1 = SocialAgent("Agent_1", personality="friendly", interests=["AI"])
        optimizer.store_agent(agent1)

        # Verify agent is in cache
        assert agent1.agent_id in optimizer.agent_cache

        # Store same agent with different data to trigger cache update
        agent1_updated = SocialAgent("Agent_1", personality="analytical", interests=["ML"])
        optimizer.store_agent(agent1_updated)

        # Verify cache was updated (deleted and re-added)
        assert agent1.agent_id in optimizer.agent_cache
        cached_agent = optimizer.agent_cache[agent1.agent_id]
        assert cached_agent.personality == "analytical"
        assert "ML" in cached_agent.interests

    def test_recompression_failure_keeps_original_data(self):
        """Test that recompression failure keeps original compressed data"""
        from src.agents.memory_optimizer import MemoryOptimizer
        from src.agents.social_agent import SocialAgent
        import unittest.mock

        optimizer = MemoryOptimizer(max_agents=10, compression_enabled=True)

        # Store an agent
        agent = SocialAgent("Test_Agent", personality="friendly", interests=["AI"])
        optimizer.store_agent(agent)

        # Get the original compressed data
        original_compressed = optimizer.agent_storage[agent.agent_id]

        # Mock compress_agent to fail during compaction
        def failing_compress(agent_obj):
            raise Exception("Recompression failed")

        with unittest.mock.patch.object(optimizer, 'compress_agent', side_effect=failing_compress):
            # Run compaction - should handle recompression failure gracefully
            result = optimizer.compact_memory()

        # Should still have the original data
        assert agent.agent_id in optimizer.agent_storage
        assert optimizer.agent_storage[agent.agent_id] == original_compressed

        # Agent should still be retrievable
        retrieved = optimizer.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id

    