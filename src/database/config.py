"""
Database configuration management
"""

import os
from typing import Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base

try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


class DatabaseConfig:
    """Database configuration class"""

    def __init__(self):
        self.settings = Settings()

    @property
    def database_url(self) -> str:
        """Get database URL from environment or settings"""
        # Priority: Environment variable > Settings > Default SQLite
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url

        # Use settings if available
        if hasattr(self.settings, 'database_url') and self.settings.database_url:
            return self.settings.database_url

        # Default to SQLite for development
        return 'sqlite:///million_agents.db'

    @property
    def echo(self) -> bool:
        """Whether to echo SQL statements"""
        return os.getenv('DATABASE_ECHO', 'false').lower() == 'true'

    @property
    def pool_size(self) -> int:
        """Database connection pool size"""
        return int(os.getenv('DATABASE_POOL_SIZE', '10'))

    @property
    def max_overflow(self) -> int:
        """Maximum overflow connections"""
        return int(os.getenv('DATABASE_MAX_OVERFLOW', '20'))


# Global database config instance
db_config = DatabaseConfig()

# Declarative base for all models
Base = declarative_base()

# Metadata for migrations
metadata = Base.metadata


def get_database_url() -> str:
    """Get the database URL"""
    return db_config.database_url


def create_engine(**kwargs):
    """Create database engine with configuration"""
    engine_kwargs = {
        'echo': db_config.echo,
        'pool_size': db_config.pool_size,
        'max_overflow': db_config.max_overflow,
    }

    # SQLite specific settings
    if db_config.database_url.startswith('sqlite'):
        engine_kwargs.update({
            'connect_args': {'check_same_thread': False},
            'poolclass': None,  # SQLite doesn't support connection pooling
        })

    # Merge with any additional kwargs
    engine_kwargs.update(kwargs)

    from sqlalchemy import create_engine as sa_create_engine
    return sa_create_engine(db_config.database_url, **engine_kwargs)