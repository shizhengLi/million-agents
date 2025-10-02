"""
Database configuration tests
"""

import pytest
import os
from unittest.mock import patch
from sqlalchemy import create_engine


class TestDatabaseConfig:
    """Test database configuration"""

    def test_database_config_creation(self):
        """Test database config object creation"""
        from src.database.config import DatabaseConfig

        config = DatabaseConfig()

        assert hasattr(config, 'database_url')
        assert hasattr(config, 'echo')
        assert hasattr(config, 'pool_size')
        assert hasattr(config, 'max_overflow')

    def test_default_sqlite_url(self):
        """Test default SQLite URL"""
        from src.database.config import DatabaseConfig

        config = DatabaseConfig()

        # Should default to SQLite when no env var is set
        assert config.database_url == 'sqlite:///million_agents.db'

    def test_environment_variable_priority(self):
        """Test that environment variables take priority"""
        from src.database.config import DatabaseConfig

        # Mock environment variable
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test@localhost/test'}):
            config = DatabaseConfig()
            assert config.database_url == 'postgresql://test@localhost/test'

    def test_echo_configuration(self):
        """Test echo configuration"""
        from src.database.config import DatabaseConfig

        # Test default
        config = DatabaseConfig()
        assert config.echo is False

        # Test with environment variable
        with patch.dict(os.environ, {'DATABASE_ECHO': 'true'}):
            config = DatabaseConfig()
            assert config.echo is True

        # Test with various cases
        with patch.dict(os.environ, {'DATABASE_ECHO': 'FALSE'}):
            config = DatabaseConfig()
            assert config.echo is False

    def test_pool_size_configuration(self):
        """Test pool size configuration"""
        from src.database.config import DatabaseConfig

        # Test default
        config = DatabaseConfig()
        assert config.pool_size == 10

        # Test with environment variable
        with patch.dict(os.environ, {'DATABASE_POOL_SIZE': '20'}):
            config = DatabaseConfig()
            assert config.pool_size == 20

    def test_max_overflow_configuration(self):
        """Test max overflow configuration"""
        from src.database.config import DatabaseConfig

        # Test default
        config = DatabaseConfig()
        assert config.max_overflow == 20

        # Test with environment variable
        with patch.dict(os.environ, {'DATABASE_MAX_OVERFLOW': '30'}):
            config = DatabaseConfig()
            assert config.max_overflow == 30

    def test_get_database_url_function(self):
        """Test get_database_url function"""
        from src.database.config import get_database_url

        # Should return the same as config.database_url
        assert get_database_url() == 'sqlite:///million_agents.db'

    def test_create_engine_sqlite(self):
        """Test engine creation for SQLite"""
        from src.database.config import create_engine

        engine = create_engine()

        assert engine is not None
        assert engine.url.drivername == 'sqlite'
        assert engine.echo is False

    def test_create_engine_postgresql(self):
        """Test engine creation for PostgreSQL (skip if psycopg2 not available)"""
        from src.database.config import create_engine

        try:
            with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test@localhost/test'}):
                engine = create_engine()
                assert engine is not None
                assert engine.url.drivername == 'postgresql'
        except ModuleNotFoundError:
            # Skip test if psycopg2 is not available
            pytest.skip("psycopg2 not available, skipping PostgreSQL test")

    def test_engine_kwargs_merge(self):
        """Test that additional kwargs are merged with defaults"""
        from src.database.config import create_engine

        with patch.dict(os.environ, {'DATABASE_ECHO': 'true'}):
            engine = create_echo_true = create_engine()
            assert create_echo_true.echo is True

            # Override echo with kwarg
            engine_echo_false = create_engine(echo=False)
            assert engine_echo_false.echo is False

    def test_base_model_available(self):
        """Test that Base model is available"""
        from src.database.config import Base

        assert Base is not None
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')

    def test_metadata_available(self):
        """Test that metadata is available"""
        from src.database.config import metadata

        assert metadata is not None
        assert hasattr(metadata, 'reflect')
        assert hasattr(metadata, 'create_all')