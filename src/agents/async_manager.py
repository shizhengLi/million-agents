"""
Async Agent Manager for high-performance concurrent operations
Supports million-scale agent populations with async processing
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
import concurrent.futures
from collections import defaultdict

from .social_agent import SocialAgent
from .batch_manager import BatchAgentManager
from .memory_optimizer import MemoryOptimizer


@dataclass
class AsyncPerformanceMetrics:
    """Performance metrics for async operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_time: float = 0.0
    average_operation_time: float = 0.0
    peak_concurrent_operations: int = 0
    operations_per_second: float = 0.0


@dataclass
class AsyncConfig:
    """Configuration for async operations"""
    max_concurrent: int = 10
    batch_size: int = 50
    rate_limit_delay: float = 0.1  # seconds between API calls
    timeout: float = 30.0  # seconds
    auto_optimize: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


class AsyncAgentManager:
    """High-performance async agent manager for million-scale operations"""

    def __init__(
        self,
        config: Optional[AsyncConfig] = None,
        max_concurrent: Optional[int] = None,
        batch_size: Optional[int] = None,
        auto_optimize: Optional[bool] = None
    ):
        """Initialize async agent manager

        Args:
            config: Async configuration settings
            max_concurrent: Maximum concurrent operations (overrides config)
            batch_size: Default batch size (overrides config)
            auto_optimize: Enable auto-optimization (overrides config)
        """
        # Create config with parameter overrides
        if config is None:
            config = AsyncConfig()

        if max_concurrent is not None:
            config.max_concurrent = max_concurrent
        if batch_size is not None:
            config.batch_size = batch_size
        if auto_optimize is not None:
            config.auto_optimize = auto_optimize

        self.config = config
        self.batch_manager = BatchAgentManager()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.metrics = AsyncPerformanceMetrics()
        self._current_batch_size = self.config.batch_size
        self._operation_times = []
        self._active_operations = 0
        self._peak_active_operations = 0
        self._last_optimization_time = time.time()
        self.logger = logging.getLogger(__name__)

    async def create_agents_async(
        self,
        count: int,
        name_prefix: str = "AsyncAgent",
        personalities: Optional[List[str]] = None,
        interests_list: Optional[List[List[str]]] = None
    ) -> List[SocialAgent]:
        """Create multiple agents asynchronously

        Args:
            count: Number of agents to create
            name_prefix: Prefix for agent names
            personalities: List of personalities to assign
            interests_list: List of interest lists for agents

        Returns:
            List of created agents
        """
        start_time = time.time()

        # Create agents in batches
        batch_size = min(self._current_batch_size, count)
        tasks = []

        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            task = self._create_agent_batch(
                batch_count=batch_count,
                name_prefix=name_prefix,
                personalities=personalities,
                interests_list=interests_list,
                batch_index=i
            )
            tasks.append(task)

        # Execute batches concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        agents = []
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch creation failed: {result}")
                self.metrics.failed_operations += 1
            else:
                agents.extend(result)
                self.metrics.successful_operations += 1

        # Update metrics
        total_time = time.time() - start_time
        self._update_metrics(total_time, len(agents))

        # Optimize batch size if enabled
        if self.config.auto_optimize:
            await self._optimize_batch_size(total_time, len(agents))

        return agents

    async def _create_agent_batch(
        self,
        batch_count: int,
        name_prefix: str,
        personalities: Optional[List[str]],
        interests_list: Optional[List[List[str]]],
        batch_index: int
    ) -> List[SocialAgent]:
        """Create a batch of agents with concurrency control"""
        async with self.semaphore:
            self._active_operations += 1
            self._peak_active_operations = max(
                self._peak_active_operations, self._active_operations
            )

            try:
                # Create agents using the existing batch manager
                agents = self.batch_manager.create_batch_agents(
                    count=batch_count,
                    name_prefix=name_prefix,
                    personalities=personalities,
                    interests_list=interests_list
                )

                # Simulate async work (in real implementation, this might involve
                # async database operations, API calls, etc.)
                await asyncio.sleep(0.001)  # Small delay to simulate async work

                return agents

            finally:
                self._active_operations -= 1

    async def batch_generate_interactions(
        self,
        agents: List[SocialAgent],
        context: str,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate interactions from multiple agents concurrently

        Args:
            agents: List of agents to generate interactions from
            context: Context for message generation
            max_concurrent: Maximum concurrent operations

        Returns:
            List of interaction results
        """
        if not agents:
            return []

        max_concurrent = max_concurrent or self.config.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_single_interaction(agent: SocialAgent) -> Dict[str, Any]:
            async with semaphore:
                self._active_operations += 1
                self._peak_active_operations = max(
                    self._peak_active_operations, self._active_operations
                )

                start_time = time.time()
                try:
                    # Rate limiting
                    await asyncio.sleep(self.config.rate_limit_delay)

                    # Generate message using existing sync method
                    # In a fully async implementation, this would use async OpenAI client
                    message = agent.generate_message(context)

                    return {
                        'agent_id': agent.agent_id,
                        'agent_name': agent.name,
                        'personality': agent.personality,
                        'message': message,
                        'timestamp': time.time(),
                        'context': context
                    }

                except Exception as e:
                    self.logger.error(f"Failed to generate interaction for {agent.agent_id}: {e}")
                    return {
                        'agent_id': agent.agent_id,
                        'agent_name': agent.name,
                        'error': str(e),
                        'timestamp': time.time()
                    }
                finally:
                    self._active_operations -= 1
                    operation_time = time.time() - start_time
                    self._operation_times.append(operation_time)

        # Create tasks for all agents
        tasks = [generate_single_interaction(agent) for agent in agents]

        # Execute with timeout
        try:
            interactions = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Interaction generation timed out")
            return []

        # Filter exceptions and update metrics
        valid_interactions = []
        for result in interactions:
            if isinstance(result, Exception):
                self.metrics.failed_operations += 1
            else:
                valid_interactions.append(result)
                self.metrics.successful_operations += 1

        return valid_interactions

    async def build_friendships_async(
        self,
        agents: List[SocialAgent],
        max_friends_per_agent: int = 5,
        batch_size: Optional[int] = None
    ) -> int:
        """Build friendships between agents asynchronously

        Args:
            agents: List of agents to build friendships for
            max_friends_per_agent: Maximum friends per agent
            batch_size: Batch size for processing

        Returns:
            Number of friendships created
        """
        if not agents or len(agents) < 2:
            return 0

        batch_size = batch_size or self._current_batch_size
        friendships_created = 0

        # Process agents in batches
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]

            async with self.semaphore:
                # We need to temporarily add batch agents to the batch manager
                # since it works with its internal agent list
                original_agents = list(self.batch_manager.agents.values())

                # Clear and add current batch
                self.batch_manager.agents.clear()
                self.batch_manager.next_id = 0
                for agent in batch:
                    self.batch_manager.agents[agent.agent_id] = agent
                    self.batch_manager.next_id = max(self.batch_manager.next_id, int(agent.agent_id.split('_')[1]) + 1)

                # Create friendships for this batch
                batch_friendships = self.batch_manager.create_random_friendships(
                    max_friends_per_agent=max_friends_per_agent
                )
                friendships_created += batch_friendships

                # Restore original agents
                self.batch_manager.agents.clear()
                for agent in original_agents:
                    self.batch_manager.agents[agent.agent_id] = agent
                if original_agents:
                    self.batch_manager.next_id = max(int(agent.agent_id.split('_')[1]) for agent in original_agents) + 1

                # Simulate async processing
                await asyncio.sleep(0.001)

        return friendships_created

    async def create_communities_async(
        self,
        agents: List[SocialAgent],
        community_names: List[str],
        max_members_per_community: int
    ) -> Dict[str, List[SocialAgent]]:
        """Create communities and assign members asynchronously

        Args:
            agents: List of agents to assign to communities
            community_names: Names of communities to create
            max_members_per_community: Maximum members per community

        Returns:
            Dictionary mapping community names to member lists
        """
        if not agents or not community_names:
            return {}

        async with self.semaphore:
            # Use existing batch manager for community creation
            communities = self.batch_manager.create_communities(
                community_names=community_names,
                max_members_per_community=max_members_per_community
            )

            # Simulate async processing
            await asyncio.sleep(0.001)

            return communities

    async def store_agents_async(
        self,
        agents: List[SocialAgent],
        memory_optimizer: MemoryOptimizer
    ) -> None:
        """Store agents in memory optimizer asynchronously

        Args:
            agents: List of agents to store
            memory_optimizer: Memory optimizer instance
        """
        if not agents:
            return

        batch_size = min(self._current_batch_size, len(agents))

        # Store in batches
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]

            async with self.semaphore:
                for agent in batch:
                    memory_optimizer.store_agent(agent)

                # Simulate async I/O
                await asyncio.sleep(0.0001)

    async def retrieve_agents_async(
        self,
        agent_ids: List[str],
        memory_optimizer: MemoryOptimizer
    ) -> List[SocialAgent]:
        """Retrieve agents from memory optimizer asynchronously

        Args:
            agent_ids: List of agent IDs to retrieve
            memory_optimizer: Memory optimizer instance

        Returns:
            List of retrieved agents
        """
        if not agent_ids:
            return []

        async def retrieve_single_agent(agent_id: str) -> Optional[SocialAgent]:
            async with self.semaphore:
                # Retrieve agent
                agent = memory_optimizer.get_agent(agent_id)

                # Simulate async I/O
                await asyncio.sleep(0.0001)

                return agent

        # Retrieve concurrently
        tasks = [retrieve_single_agent(agent_id) for agent_id in agent_ids]
        agents = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        valid_agents = [agent for agent in agents
                       if agent is not None and not isinstance(agent, Exception)]

        return valid_agents

    async def find_compatible_matches_async(
        self,
        agents: List[SocialAgent],
        candidate_agents: List[SocialAgent],
        threshold: float = 0.5,
        max_matches: int = 5
    ) -> Dict[str, List[str]]:
        """Find compatible matches for agents asynchronously

        Args:
            agents: Agents to find matches for
            candidate_agents: Potential match candidates
            threshold: Compatibility threshold
            max_matches: Maximum matches per agent

        Returns:
            Dictionary mapping agent IDs to lists of compatible agent IDs
        """
        matches = {}

        async def find_matches_for_agent(agent: SocialAgent) -> Tuple[str, List[str]]:
            async with self.semaphore:
                compatible = []
                for candidate in candidate_agents:
                    if agent.agent_id != candidate.agent_id:
                        compatibility = agent.calculate_compatibility(candidate)
                        if compatibility >= threshold:
                            compatible.append(candidate.agent_id)

                            if len(compatible) >= max_matches:
                                break

                # Simulate processing time
                await asyncio.sleep(0.001)

                return agent.agent_id, compatible

        # Find matches concurrently
        tasks = [find_matches_for_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compile results
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Match finding failed: {result}")
            else:
                agent_id, compatible = result
                matches[agent_id] = compatible

        return matches

    async def analyze_social_network_async(
        self,
        agents: List[SocialAgent]
    ) -> Dict[str, Any]:
        """Analyze social network properties asynchronously

        Args:
            agents: Agents to analyze

        Returns:
            Dictionary containing network statistics
        """
        if not agents:
            return {}

        async with self.semaphore:
            # Calculate network statistics
            total_connections = sum(len(agent.friends) for agent in agents)
            avg_connections = total_connections / len(agents) if agents else 0

            # Calculate network density
            max_possible_connections = len(agents) * (len(agents) - 1) / 2
            network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0

            # Find influential agents (most connections)
            sorted_agents = sorted(agents, key=lambda a: len(a.friends), reverse=True)
            influential_agents = [agent.agent_id for agent in sorted_agents[:5]]

            # Simple cluster detection (based on mutual friendships)
            clusters = self._detect_friendship_clusters(agents)

            # Simulate analysis time
            await asyncio.sleep(0.01)

            return {
                'average_connections': avg_connections,
                'network_density': network_density,
                'total_connections': total_connections,
                'influential_agents': influential_agents,
                'clusters': len(clusters),
                'agent_count': len(agents)
            }

    def _detect_friendship_clusters(self, agents: List[SocialAgent]) -> List[List[str]]:
        """Detect friendship clusters using simple connected components"""
        visited = set()
        clusters = []

        def dfs(agent_id: str, cluster: List[str]):
            if agent_id in visited:
                return
            visited.add(agent_id)
            cluster.append(agent_id)

            # Find agent and visit friends
            for agent in agents:
                if agent.agent_id == agent_id:
                    for friend_id in agent.friends:
                        dfs(friend_id, cluster)
                    break

        for agent in agents:
            if agent.agent_id not in visited:
                cluster = []
                dfs(agent.agent_id, cluster)
                if len(cluster) > 1:  # Only count clusters with multiple agents
                    clusters.append(cluster)

        return clusters

    async def generate_recommendations_async(
        self,
        agents: List[SocialAgent],
        candidate_agents: List[SocialAgent],
        recommendation_type: str = "friends",
        max_recommendations: int = 3
    ) -> Dict[str, List[str]]:
        """Generate recommendations for agents asynchronously

        Args:
            agents: Agents to generate recommendations for
            candidate_agents: Potential recommendation candidates
            recommendation_type: Type of recommendation ("friends", "communities")
            max_recommendations: Maximum recommendations per agent

        Returns:
            Dictionary mapping agent IDs to recommendation lists
        """
        recommendations = {}

        async def generate_for_agent(agent: SocialAgent) -> Tuple[str, List[str]]:
            async with self.semaphore:
                recs = []

                if recommendation_type == "friends":
                    # Recommend friends based on compatibility
                    for candidate in candidate_agents:
                        if (candidate.agent_id != agent.agent_id and
                            candidate.agent_id not in agent.friends):

                            compatibility = agent.calculate_compatibility(candidate)
                            recs.append((candidate.agent_id, compatibility))

                # Sort by compatibility and take top recommendations
                recs.sort(key=lambda x: x[1], reverse=True)
                top_recs = [agent_id for agent_id, _ in recs[:max_recommendations]]

                # Simulate processing time
                await asyncio.sleep(0.001)

                return agent.agent_id, top_recs

        # Generate recommendations concurrently
        tasks = [generate_for_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compile results
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Recommendation generation failed: {result}")
            else:
                agent_id, recs = result
                recommendations[agent_id] = recs

        return recommendations

    async def stream_interactions(
        self,
        agents: List[SocialAgent],
        context: str,
        batch_size: int = 10
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream agent interactions in real-time batches

        Args:
            agents: Agents to generate interactions from
            context: Context for message generation
            batch_size: Size of each batch

        Yields:
            Batches of interaction results
        """
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]

            # Generate interactions for this batch
            interactions = await self.batch_generate_interactions(
                agents=batch,
                context=context,
                max_concurrent=min(batch_size, self.config.max_concurrent)
            )

            yield interactions

            # Small delay between batches to prevent overwhelming
            await asyncio.sleep(0.01)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'total_operations': self.metrics.total_operations,
            'successful_operations': self.metrics.successful_operations,
            'failed_operations': self.metrics.failed_operations,
            'total_time': self.metrics.total_time,
            'average_operation_time': self.metrics.average_operation_time,
            'peak_concurrent_operations': self.metrics.peak_concurrent_operations,
            'operations_per_second': self.metrics.operations_per_second,
            'current_batch_size': self._current_batch_size,
            'active_operations': self._active_operations
        }

    def get_current_batch_size(self) -> int:
        """Get current optimized batch size"""
        return self._current_batch_size

    def _update_metrics(self, operation_time: float, operation_count: int):
        """Update performance metrics"""
        self.metrics.total_operations += operation_count
        self.metrics.total_time += operation_time

        if self.metrics.total_operations > 0:
            self.metrics.average_operation_time = (
                self.metrics.total_time / self.metrics.total_operations
            )
            self.metrics.operations_per_second = (
                self.metrics.total_operations / self.metrics.total_time
            )

    async def _optimize_batch_size(self, operation_time: float, operation_count: int):
        """Optimize batch size based on performance"""
        current_time = time.time()

        # Only optimize every 10 seconds
        if current_time - self._last_optimization_time < 10:
            return

        ops_per_second = operation_count / operation_time if operation_time > 0 else 0

        # Simple optimization: if operations are fast, increase batch size
        # if operations are slow, decrease batch size
        if ops_per_second > 1000:  # Very fast operations
            self._current_batch_size = min(self._current_batch_size * 2, 200)
        elif ops_per_second < 100:  # Slow operations
            self._current_batch_size = max(self._current_batch_size // 2, 10)

        self._last_optimization_time = current_time

    @property
    def agents(self) -> List[SocialAgent]:
        """Get all agents from the batch manager"""
        return list(self.batch_manager.agents.values())