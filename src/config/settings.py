"""
Configuration settings for million agents social application
"""

import os
from typing import Optional
from dotenv import load_dotenv


class Settings:
    """Application settings loaded from environment variables"""

    _instance: Optional['Settings'] = None

    def __new__(cls) -> 'Settings':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize settings by loading environment variables"""
        if hasattr(self, '_initialized'):
            return

        load_dotenv()
        self._load_required_vars()
        self._load_optional_vars()
        self._validate_values()
        self._initialized = True

    def _load_required_vars(self):
        """Load required environment variables"""
        self.openai_api_key = self._get_required_env('OPENAI_API_KEY')
        self.openai_base_url = self._get_required_env('OPENAI_BASE_URL')
        self.openai_model = self._get_required_env('OPENAI_MODEL')

    def _load_optional_vars(self):
        """Load optional environment variables with defaults"""
        try:
            self.max_agents = int(os.getenv('MAX_AGENTS', '1000000'))
            self.agent_batch_size = int(os.getenv('AGENT_BATCH_SIZE', '100'))
            self.interaction_interval = int(os.getenv('INTERACTION_INTERVAL', '5'))
            self.community_size_limit = int(os.getenv('COMMUNITY_SIZE_LIMIT', '1000'))
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in environment variables: {e}")

        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///social_agents.db')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def _validate_values(self):
        """Validate configuration values"""
        # Validate numeric values
        if not isinstance(self.max_agents, int) or self.max_agents <= 0:
            raise ValueError("MAX_AGENTS must be a positive integer")

        if not isinstance(self.agent_batch_size, int) or self.agent_batch_size <= 0:
            raise ValueError("AGENT_BATCH_SIZE must be a positive integer")

        if not isinstance(self.interaction_interval, int) or self.interaction_interval < 1:
            raise ValueError("INTERACTION_INTERVAL must be at least 1")

        if not isinstance(self.community_size_limit, int) or self.community_size_limit <= 0:
            raise ValueError("COMMUNITY_SIZE_LIMIT must be a positive integer")

        # Validate ranges
        if self.agent_batch_size > 1000:
            raise ValueError("AGENT_BATCH_SIZE cannot exceed 1000 for performance reasons")

        if self.agent_batch_size > self.max_agents:
            raise ValueError("AGENT_BATCH_SIZE cannot exceed MAX_AGENTS")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_log_levels)}")

    def _get_required_env(self, key: str) -> str:
        """Get required environment variable"""
        value = os.getenv(key)
        if value is None or value.strip() == '':
            raise ValueError(f"Missing required environment variable: {key}")
        return value.strip()

    def __str__(self) -> str:
        """String representation with sensitive data masked"""
        return (
            f"Settings(\n"
            f"  openai_api_key=***{self.openai_api_key[-4:] if len(self.openai_api_key) > 4 else '***'},\n"
            f"  openai_base_url={self.openai_base_url},\n"
            f"  openai_model={self.openai_model},\n"
            f"  max_agents={self.max_agents},\n"
            f"  agent_batch_size={self.agent_batch_size},\n"
            f"  interaction_interval={self.interaction_interval},\n"
            f"  community_size_limit={self.community_size_limit},\n"
            f"  database_url={self.database_url},\n"
            f"  log_level={self.log_level}\n"
            f")"
        )