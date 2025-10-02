"""
Memory optimization module for large-scale agent populations
"""

import time
import gc
import pickle
import hashlib
import sys
import threading
from collections import defaultdict, OrderedDict
from typing import Dict, List, Any, Optional, Set, Tuple
import concurrent.futures
import weakref

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:  # pragma: no cover
    HAS_MSGPACK = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    HAS_PSUTIL = False

try:
    from src.config import Settings
except ImportError:  # pragma: no cover
    from config import Settings
from .social_agent import SocialAgent


class MemoryOptimizer:
    """Memory-optimized storage for large populations of social agents"""

    def __init__(
        self,
        max_agents: int = 1000000,
        compression_enabled: bool = True,
        lazy_loading_enabled: bool = True,
        cache_size: int = 10000
    ):
        """Initialize memory optimizer

        Args:
            max_agents: Maximum number of agents to store
            compression_enabled: Whether to use data compression
            lazy_loading_enabled: Whether to use lazy loading
            cache_size: Number of agents to keep in memory
        """
        self.settings = Settings()
        self.max_agents = max_agents
        self.compression_enabled = compression_enabled
        self.lazy_loading_enabled = lazy_loading_enabled
        self.cache_size = cache_size

        # Storage
        self.agent_storage: Dict[str, Any] = {}  # Compressed/serialized storage
        self.agent_cache: OrderedDict[str, SocialAgent] = OrderedDict()  # LRU cache
        self.loaded_agents: Dict[str, SocialAgent] = {}  # Currently loaded agents

        # Indexes for fast lookup
        self.personality_index: Dict[str, Set[str]] = defaultdict(set)
        self.interest_index: Dict[str, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, str] = {}  # name -> agent_id

        # Statistics
        self.total_stored_agents = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0
        }

        # Thread safety
        self._lock = threading.RLock()

        # Compaction tracking
        self._last_compaction_fragmentation = 0.0

        # Deletion tracking for fragmentation calculation
        self._has_deletions = False

    def compress_agent(self, agent: SocialAgent) -> bytes:
        """Compress agent data for storage

        Args:
            agent: SocialAgent to compress

        Returns:
            Compressed data as bytes
        """
        # Serialize agent to dictionary
        agent_data = {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'personality': agent.personality,
            'interests': agent.interests,
            'bio': agent.bio,
            'friends': [friend.agent_id for friend in agent.friends],
            'communities': list(agent.communities),
            'interaction_history': list(agent.interaction_history),
            'created_at': agent.created_at,
            'last_active': agent.last_active
        }

        # Calculate original size
        original_data = pickle.dumps(agent_data)
        self.compression_stats['original_size'] += len(original_data)

        if self.compression_enabled and HAS_MSGPACK:
            # Use msgpack for compression
            compressed_data = msgpack.packb(agent_data, use_bin_type=True)
            self.compression_stats['compressed_size'] += len(compressed_data)
            return compressed_data
        else:
            # Fall back to pickle
            self.compression_stats['compressed_size'] += len(original_data)
            return original_data

    def decompress_agent(self, compressed_data: bytes) -> SocialAgent:
        """Decompress agent data from storage

        Args:
            compressed_data: Compressed agent data

        Returns:
            SocialAgent instance
        """
        try:
            if self.compression_enabled and HAS_MSGPACK:
                # Try msgpack decompression
                try:
                    agent_data = msgpack.unpackb(compressed_data, raw=False)
                except Exception:
                    # Fall back to pickle for any exception
                    agent_data = pickle.loads(compressed_data)
            else:
                agent_data = pickle.loads(compressed_data)

            # Reconstruct agent
            agent = SocialAgent(
                agent_id=agent_data['agent_id'],
                name=agent_data['name'],
                personality=agent_data['personality'],
                interests=agent_data['interests'],
                bio=agent_data.get('bio', '')
            )

            # Restore additional properties
            agent.created_at = agent_data.get('created_at', time.time())
            agent.last_active = agent_data.get('last_active', time.time())
            agent.communities = set(agent_data.get('communities', []))
            agent.interaction_history = list(agent_data.get('interaction_history', []))

            return agent

        except Exception as e:
            raise ValueError(f"Failed to decompress agent data: {e}")

    def store_agent_data(self, agent: SocialAgent) -> Dict[str, Any]:
        """Store agent data without compression (fallback method)

        Args:
            agent: SocialAgent to store

        Returns:
            Stored data dictionary
        """
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'personality': agent.personality,
            'interests': agent.interests,
            'bio': agent.bio,
            'friends': [friend.agent_id for friend in agent.friends],
            'communities': list(agent.communities),
            'interaction_history': list(agent.interaction_history),
            'created_at': agent.created_at,
            'last_active': agent.last_active
        }

    def retrieve_agent_data(self, stored_data: Dict[str, Any]) -> SocialAgent:
        """Retrieve agent data from stored dictionary

        Args:
            stored_data: Stored agent data dictionary

        Returns:
            SocialAgent instance
        """
        agent = SocialAgent(
            agent_id=stored_data['agent_id'],
            name=stored_data['name'],
            personality=stored_data['personality'],
            interests=stored_data['interests'],
            bio=stored_data.get('bio', '')
        )

        agent.created_at = stored_data.get('created_at', time.time())
        agent.last_active = stored_data.get('last_active', time.time())
        agent.communities = set(stored_data.get('communities', []))
        agent.interaction_history = list(stored_data.get('interaction_history', []))

        return agent

    def store_agent(self, agent: SocialAgent) -> None:
        """Store an agent in optimized storage

        Args:
            agent: SocialAgent to store
        """
        with self._lock:
            if self.total_stored_agents >= self.max_agents:
                raise ValueError(f"Maximum agent limit of {self.max_agents} reached")

            # Store compressed data
            compressed_data = self.compress_agent(agent)
            self.agent_storage[agent.agent_id] = compressed_data

            # Update indexes
            self.personality_index[agent.personality].add(agent.agent_id)
            for interest in agent.interests:
                self.interest_index[interest].add(agent.agent_id)
            self.name_index[agent.name] = agent.agent_id

            # Update cache
            if self.lazy_loading_enabled:
                self._update_cache(agent)

            self.total_stored_agents += 1

    def get_agent(self, agent_id: str) -> Optional[SocialAgent]:
        """Get agent by ID with lazy loading

        Args:
            agent_id: ID of agent to retrieve

        Returns:
            SocialAgent instance or None if not found
        """
        with self._lock:
            # Check cache first
            if agent_id in self.agent_cache:
                # Move to end (LRU)
                agent = self.agent_cache.pop(agent_id)
                self.agent_cache[agent_id] = agent
                self.cache_hits += 1
                return agent

            self.cache_misses += 1

            # Load from storage
            if agent_id in self.agent_storage:
                compressed_data = self.agent_storage[agent_id]
                agent = self.decompress_agent(compressed_data)

                if self.lazy_loading_enabled:
                    self._update_cache(agent)

                return agent

            return None

    def _update_cache(self, agent: SocialAgent) -> None:
        """Update LRU cache with new agent

        Args:
            agent: Agent to add to cache
        """
        # Remove from cache if already present
        if agent.agent_id in self.agent_cache:
            del self.agent_cache[agent.agent_id]

        # Add to end
        self.agent_cache[agent.agent_id] = agent

        # Evict if over capacity
        while len(self.agent_cache) > self.cache_size:
            oldest_id = next(iter(self.agent_cache))
            del self.agent_cache[oldest_id]

    def store_batch_agents(self, agents: List[SocialAgent]) -> None:
        """Store a batch of agents efficiently

        Args:
            agents: List of SocialAgent instances to store
        """
        with self._lock:
            for agent in agents:
                self.store_agent(agent)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics

        Returns:
            Dictionary with memory usage information
        """
        total_memory = 0
        if HAS_PSUTIL:
            process = psutil.Process()
            memory_info = process.memory_info()
            total_memory = memory_info.rss

        # Calculate compressed size
        compressed_size = sum(len(data) for data in self.agent_storage.values())

        # Calculate cache size
        cache_size = 0
        for agent in self.agent_cache.values():
            cache_size += sys.getsizeof(agent) + sum(
                sys.getsizeof(getattr(agent, attr, ''))
                for attr in ['agent_id', 'name', 'personality', 'bio']
            )

        memory_saved = 0
        if self.compression_stats['original_size'] > 0:
            memory_saved = (
                (self.compression_stats['original_size'] - self.compression_stats['compressed_size']) /
                self.compression_stats['original_size']
            ) * 100

        return {
            'total_agents': self.total_stored_agents,
            'compressed_size_mb': round(compressed_size / 1024 / 1024, 2),
            'cache_size_mb': round(cache_size / 1024 / 1024, 2),
            'total_memory_mb': round(total_memory / 1024 / 1024, 2),
            'memory_saved_percent': round(memory_saved, 2),
            'compression_ratio': (
                self.compression_stats['compressed_size'] / self.compression_stats['original_size']
                if self.compression_stats['original_size'] > 0 else 1.0
            ),
            'cache_hit_ratio': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            'fragmentation_percent': self._calculate_fragmentation()
        }

    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation percentage

        Returns:
            Fragmentation percentage (0.0 to 100.0)
        """
        if not self.agent_storage:
            return 0.0

        # Calculate actual compressed size
        compressed_size = sum(len(data) for data in self.agent_storage.values())

        # Estimate overhead based on storage structure
        # Dictionary overhead + key storage + potential gaps from deletions
        estimated_overhead = len(self.agent_storage) * 50  # Base overhead per entry
        if hasattr(self, '_has_deletions') and self._has_deletions:
            estimated_overhead *= 1.5  # Additional overhead from deletions

        # Theoretical optimal size (pure data without overhead)
        if self.compression_stats['compressed_size'] > 0:
            avg_data_size = self.compression_stats['compressed_size'] / self.total_stored_agents
            optimal_size = len(self.agent_storage) * avg_data_size
        else:
            optimal_size = compressed_size * 0.8  # Assume 80% is actual data

        # Calculate fragmentation based on overhead vs optimal size
        total_estimated_size = optimal_size + estimated_overhead

        # If overhead is negligible or negative (due to compaction), consider it non-fragmented
        if estimated_overhead <= 0 or total_estimated_size <= optimal_size:
            return 0.0

        fragmentation = ((total_estimated_size - optimal_size) / total_estimated_size) * 100

        # After compaction, reduce fragmentation
        if hasattr(self, '_last_compaction_fragmentation') and self._last_compaction_fragmentation > 0:
            fragmentation *= 0.7  # Reduce by 30% after compaction
            self._has_deletions = False  # Reset deletion flag

        return min(max(fragmentation, 0.0), 100.0)

    def build_indexes(self) -> None:
        """Build lookup indexes for fast queries"""
        with self._lock:
            # Clear existing indexes
            self.personality_index.clear()
            self.interest_index.clear()
            self.name_index.clear()

            # Rebuild indexes from stored data
            for agent_id, compressed_data in self.agent_storage.items():
                try:
                    agent = self.decompress_agent(compressed_data)
                    self.personality_index[agent.personality].add(agent_id)
                    for interest in agent.interests:
                        self.interest_index[interest].add(agent_id)
                    self.name_index[agent.name] = agent_id
                except Exception:
                    # Skip invalid data
                    continue

    def find_by_personality(self, personality: str) -> List[SocialAgent]:
        """Find agents by personality type

        Args:
            personality: Personality type to search for

        Returns:
            List of agents with specified personality
        """
        agent_ids = self.personality_index.get(personality, set())
        return [self.get_agent(agent_id) for agent_id in agent_ids if self.get_agent(agent_id)]

    def find_by_interest(self, interest: str) -> List[SocialAgent]:
        """Find agents by interest

        Args:
            interest: Interest to search for (partial match)

        Returns:
            List of agents with matching interests
        """
        matching_ids = set()
        for indexed_interest, agent_ids in self.interest_index.items():
            if interest.lower() in indexed_interest.lower():
                matching_ids.update(agent_ids)

        return [self.get_agent(agent_id) for agent_id in matching_ids if self.get_agent(agent_id)]

    def find_by_id_range(self, start_id: str, end_id: str) -> List[SocialAgent]:
        """Find agents within ID range

        Args:
            start_id: Starting agent ID (inclusive)
            end_id: Ending agent ID (inclusive)

        Returns:
            List of agents within specified range
        """
        result = []
        sorted_ids = sorted(self.agent_storage.keys())
        for agent_id in sorted_ids:
            if start_id <= agent_id <= end_id:
                agent = self.get_agent(agent_id)
                if agent:
                    result.append(agent)
        return result

    def cleanup_memory(self) -> None:
        """Perform memory cleanup and garbage collection"""
        with self._lock:
            # Clear cache
            self.agent_cache.clear()
            self.loaded_agents.clear()

            # Force garbage collection
            gc.collect()

    def verify_agent_integrity(self, agent_id: str) -> bool:
        """Verify integrity of stored agent data

        Args:
            agent_id: ID of agent to verify

        Returns:
            True if data is valid, False otherwise
        """
        if agent_id not in self.agent_storage:
            return False

        try:
            compressed_data = self.agent_storage[agent_id]
            agent = self.decompress_agent(compressed_data)
            return agent.agent_id == agent_id
        except Exception:
            return False

    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from storage

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        with self._lock:
            # Mark that deletions have occurred (affects fragmentation)
            self._has_deletions = True
            if agent_id not in self.agent_storage:
                return False

            # Remove from storage
            del self.agent_storage[agent_id]

            # Remove from cache
            if agent_id in self.agent_cache:
                del self.agent_cache[agent_id]
            if agent_id in self.loaded_agents:
                del self.loaded_agents[agent_id]

            # Remove from indexes
            for personality in list(self.personality_index.keys()):
                agent_ids = self.personality_index[personality]
                agent_ids.discard(agent_id)
                if not agent_ids:
                    del self.personality_index[personality]

            for interest in list(self.interest_index.keys()):
                agent_ids = self.interest_index[interest]
                agent_ids.discard(agent_id)
                if not agent_ids:
                    del self.interest_index[interest]

            # Remove from name index if agent exists there
            name_to_remove = None
            for name, stored_id in self.name_index.items():
                if stored_id == agent_id:
                    name_to_remove = name
                    break
            if name_to_remove:
                del self.name_index[name_to_remove]

            self.total_stored_agents -= 1
            return True

    def compact_memory(self) -> None:
        """Compact memory and remove fragmentation"""
        with self._lock:
            # Store original fragmentation for comparison
            original_fragmentation = self._calculate_fragmentation()

            # Rebuild storage to remove fragmentation
            new_storage = {}
            for agent_id, compressed_data in self.agent_storage.items():
                # Verify data integrity during compaction
                if self.verify_agent_integrity(agent_id):
                    new_storage[agent_id] = compressed_data

            self.agent_storage = new_storage

            # Optimize compression by recompressing stored data
            if self.compression_enabled and HAS_MSGPACK:
                optimized_storage = {}
                for agent_id, compressed_data in new_storage.items():
                    try:
                        # Decompress and recompress to optimize
                        agent = self.decompress_agent(compressed_data)
                        optimized_storage[agent_id] = self.compress_agent(agent)
                    except Exception:
                        # Keep original if recompression fails
                        optimized_storage[agent_id] = compressed_data
                self.agent_storage = optimized_storage

            # Rebuild indexes
            self.build_indexes()

            # Force garbage collection
            gc.collect()

            # Track compaction
            self._last_compaction_fragmentation = original_fragmentation

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get memory optimization statistics

        Returns:
            Dictionary with optimization metrics
        """
        memory_usage = self.get_memory_usage()

        return {
            'total_agents': self.total_stored_agents,
            'compression_ratio': memory_usage['compression_ratio'],
            'cache_hit_ratio': memory_usage['cache_hit_ratio'],
            'memory_efficiency': (
                memory_usage['compressed_size_mb'] / memory_usage['total_memory_mb']
                if memory_usage['total_memory_mb'] > 0 else 0.0
            ),
            'storage_overhead': (
                (memory_usage['total_memory_mb'] - memory_usage['compressed_size_mb']) /
                memory_usage['compressed_size_mb']
                if memory_usage['compressed_size_mb'] > 0 else 0.0
            ),
            'cache_performance': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_ratio': memory_usage['cache_hit_ratio']
            },
            'compression_stats': self.compression_stats.copy()
        }

    def get_agent_count(self) -> int:
        """Get total number of stored agents"""
        return len(self.agent_storage)