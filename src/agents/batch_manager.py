"""
Batch agent manager for creating and managing large populations of social agents
"""

import random
import time
import json
import sys
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter
import psutil
import os

try:
    from src.config import Settings
except ImportError:  # pragma: no cover
    from config import Settings
from .social_agent import SocialAgent


class BatchAgentManager:
    """Manager for creating and handling large populations of social agents"""

    # Default personality types for batch creation
    DEFAULT_PERSONALITIES = ["friendly", "analytical", "creative", "formal", "casual"]

    # Default interest pools for batch creation
    DEFAULT_INTEREST_POOLS = [
        ["AI", "machine learning", "research"],
        ["data science", "statistics", "analytics"],
        ["art", "design", "creativity"],
        ["social networks", "communication", "community"],
        ["programming", "software", "technology"],
        ["science", "biology", "medicine"],
        ["business", "entrepreneurship", "innovation"],
        ["education", "teaching", "learning"]
    ]

    def __init__(self, max_agents: int = 1000000, batch_size: int = 100):
        """Initialize batch agent manager

        Args:
            max_agents: Maximum number of agents allowed
            batch_size: Default batch size for operations
        """
        self.settings = Settings()
        self.max_agents = max_agents
        self.batch_size = batch_size
        self.agents: Dict[str, SocialAgent] = {}
        self.next_id = 0
        self._creation_start_time = None

    def create_agent(
        self,
        name: Optional[str] = None,
        personality: Optional[str] = None,
        interests: Optional[List[str]] = None,
        bio: Optional[str] = None
    ) -> SocialAgent:
        """Create a single social agent

        Args:
            name: Agent name (auto-generated if None)
            personality: Agent personality (random if None)
            interests: List of interests (random if None)
            bio: Agent biography (generated if None)

        Returns:
            Created SocialAgent instance

        Raises:
            ValueError: If maximum agent limit would be exceeded
        """
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Cannot create agent: maximum limit of {self.max_agents} agents reached")

        agent_id = f"agent_{self.next_id}"
        self.next_id += 1

        # Generate default values if not provided
        if name is None:
            name = f"Agent_{self.next_id - 1}"

        if personality is None:
            personality = random.choice(self.DEFAULT_PERSONALITIES)

        if interests is None:
            interests = random.choice(self.DEFAULT_INTEREST_POOLS).copy()

        if bio is None:
            bio = f"A {personality} agent interested in {', '.join(interests[:2])}"

        # Create and store agent
        agent = SocialAgent(
            agent_id=agent_id,
            name=name,
            personality=personality,
            interests=interests,
            bio=bio
        )

        self.agents[agent_id] = agent
        return agent

    def create_batch_agents(
        self,
        count: int,
        name_prefix: str = "Agent",
        personalities: Optional[List[str]] = None,
        interests_list: Optional[List[List[str]]] = None
    ) -> List[SocialAgent]:
        """Create a batch of social agents

        Args:
            count: Number of agents to create
            name_prefix: Prefix for agent names
            personalities: List of personalities to choose from (default if None)
            interests_list: List of interest pools to choose from (default if None)

        Returns:
            List of created SocialAgent instances

        Raises:
            ValueError: If count would exceed maximum agent limit
        """
        if len(self.agents) + count > self.max_agents:
            raise ValueError(
                f"Cannot create {count} agents: would exceed maximum limit of {self.max_agents} agents"
            )

        self._creation_start_time = time.time()
        agents = []

        # Use provided lists or defaults
        if personalities is None:
            personalities = self.DEFAULT_PERSONALITIES
        if interests_list is None:
            interests_list = self.DEFAULT_INTEREST_POOLS

        for i in range(count):
            agent = self.create_agent(
                name=f"{name_prefix}_{i}",
                personality=random.choice(personalities),
                interests=random.choice(interests_list).copy()
            )
            agents.append(agent)

        return agents

    def get_agent_by_id(self, agent_id: str) -> Optional[SocialAgent]:
        """Get an agent by their ID

        Args:
            agent_id: The agent ID to retrieve

        Returns:
            SocialAgent instance if found, None otherwise
        """
        return self.agents.get(agent_id)

    def get_agents_by_personality(self, personality: str) -> List[SocialAgent]:
        """Get all agents with a specific personality

        Args:
            personality: The personality to filter by

        Returns:
            List of agents with the specified personality
        """
        return [agent for agent in self.agents.values() if agent.personality == personality]

    def get_agents_by_interest(self, interest: str) -> List[SocialAgent]:
        """Get all agents with a specific interest

        Args:
            interest: The interest to filter by (partial match)

        Returns:
            List of agents with the specified interest
        """
        return [
            agent for agent in self.agents.values()
            if any(interest.lower() in agent_interest.lower() for agent_interest in agent.interests)
        ]

    def create_random_friendships(self, max_friends_per_agent: int = 5) -> int:
        """Create random friendships between agents

        Args:
            max_friends_per_agent: Maximum number of friends each agent can have

        Returns:
            Number of friendships created
        """
        if len(self.agents) < 2:
            return 0

        friendships_created = 0
        agent_list = list(self.agents.values())

        for agent in agent_list:
            if len(agent.friends) >= max_friends_per_agent:
                continue

            # Find potential friends (not already friends and not self)
            potential_friends = [
                other for other in agent_list
                if other.agent_id != agent.agent_id and other not in agent.friends
            ]

            if potential_friends:
                # Add 1-3 random friends
                num_friends = min(
                    random.randint(1, 3),
                    max_friends_per_agent - len(agent.friends),
                    len(potential_friends)
                )

                friends_to_add = random.sample(potential_friends, num_friends)
                for friend in friends_to_add:
                    agent.add_friend(friend)
                    friendships_created += 1

        return friendships_created

    def create_communities(
        self,
        community_names: List[str],
        max_members_per_community: Optional[int] = None
    ) -> Dict[str, List[SocialAgent]]:
        """Create communities and assign agents to them

        Args:
            community_names: List of community names
            max_members_per_community: Maximum members per community (no limit if None)

        Returns:
            Dictionary mapping community names to lists of member agents
        """
        if max_members_per_community is None:
            max_members_per_community = len(self.agents) // len(community_names) + 1

        communities = {name: [] for name in community_names}
        agent_list = list(self.agents.values())
        random.shuffle(agent_list)

        for agent in agent_list:
            # Assign agent to 1-2 random communities
            num_communities = random.randint(1, 2)
            available_communities = [
                name for name, members in communities.items()
                if len(members) < max_members_per_community
            ]

            if available_communities:
                assigned_communities = random.sample(
                    available_communities,
                    min(num_communities, len(available_communities))
                )

                for community_name in assigned_communities:
                    agent.join_community(community_name)
                    communities[community_name].append(agent)

        return communities

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent population

        Returns:
            Dictionary containing population statistics
        """
        if not self.agents:
            return {
                'total_agents': 0,
                'personalities': {},
                'common_interests': {},
                'average_friends': 0,
                'agents_in_communities': 0
            }

        # Personality distribution
        personalities = Counter(agent.personality for agent in self.agents.values())

        # Interest distribution
        all_interests = []
        for agent in self.agents.values():
            all_interests.extend(agent.interests)
        common_interests = Counter(all_interests).most_common(10)

        # Friendship statistics
        total_friends = sum(len(agent.friends) for agent in self.agents.values())
        average_friends = total_friends / len(self.agents) if self.agents else 0

        # Community participation
        agents_in_communities = sum(
            1 for agent in self.agents.values() if agent.communities
        )

        return {
            'total_agents': len(self.agents),
            'personalities': dict(personalities),
            'common_interests': dict(common_interests),
            'average_friends': average_friends,
            'agents_in_communities': agents_in_communities,
            'total_communities': len(set(
                community for agent in self.agents.values()
                for community in agent.communities
            ))
        }

    def run_batch_interactions(
        self,
        context: str = "General discussion",
        max_interactions: int = 100
    ) -> List[Dict[str, Any]]:
        """Run batch interactions between agents

        Args:
            context: Context for the interactions
            max_interactions: Maximum number of interactions to generate

        Returns:
            List of interaction records
        """
        interactions = []
        agent_list = list(self.agents.values())

        if not agent_list:
            return interactions

        for _ in range(min(max_interactions, len(agent_list))):
            # Select random agent
            agent = random.choice(agent_list)

            # Generate message
            try:
                message = agent.generate_message(context=context, max_length=100)

                interaction = {
                    'agent_id': agent.agent_id,
                    'agent_name': agent.name,
                    'personality': agent.personality,
                    'message': message,
                    'timestamp': time.time()
                }

                interactions.append(interaction)

                # Record interaction in agent's history
                agent.record_interaction(
                    with_agent_id="batch_interaction",
                    message=message,
                    interaction_type="batch_discussion"
                )

            except Exception as e:
                # Handle potential OpenAI API errors gracefully
                interactions.append({
                    'agent_id': agent.agent_id,
                    'agent_name': agent.name,
                    'personality': agent.personality,
                    'message': f"[Error generating message: {str(e)}]",
                    'timestamp': time.time(),
                    'error': True
                })

        return interactions

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for the agent population

        Returns:
            Dictionary containing memory usage statistics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Estimate memory per agent (rough calculation)
        total_memory_mb = memory_info.rss / 1024 / 1024
        memory_per_agent_kb = (total_memory_mb * 1024) / len(self.agents) if self.agents else 0

        return {
            'total_agents': len(self.agents),
            'estimated_memory_mb': round(total_memory_mb, 2),
            'memory_per_agent_kb': round(memory_per_agent_kb, 2),
            'process_memory_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'process_memory_vms_mb': round(memory_info.vms / 1024 / 1024, 2)
        }

    def export_agents(self, format: str = 'dict') -> Any:
        """Export agent data in specified format

        Args:
            format: Export format ('dict' or 'json')

        Returns:
            Agent data in specified format

        Raises:
            ValueError: If format is not supported
        """
        data = []
        for agent in self.agents.values():
            agent_data = agent.to_dict()
            agent_data.update({
                'friends_count': len(agent.friends),
                'communities_list': list(agent.communities),
                'interaction_count': len(agent.interaction_history)
            })
            data.append(agent_data)

        if format == 'dict':
            return data
        elif format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_all_agents(self) -> None:
        """Clear all agents from the manager"""
        self.agents.clear()
        self.next_id = 0
        self._creation_start_time = None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for batch operations

        Returns:
            Dictionary containing performance metrics
        """
        creation_time = 0
        if self._creation_start_time:
            creation_time = time.time() - self._creation_start_time

        agents_per_second = len(self.agents) / creation_time if creation_time > 0 else 0

        return {
            'agents_created': len(self.agents),
            'creation_time_seconds': round(creation_time, 3),
            'agents_per_second': round(agents_per_second, 2),
            'batch_size': self.batch_size,
            'max_agents': self.max_agents,
            'utilization_percent': round((len(self.agents) / self.max_agents) * 100, 2) if self.max_agents > 0 else 0
        }