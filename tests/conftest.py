"""
pytest configuration and fixtures
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    original_env = os.environ.copy()

    test_env = {
        'OPENAI_API_KEY': 'test-api-key-12345',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1',
        'OPENAI_MODEL': 'gpt-4o-mini',
        'MAX_AGENTS': '1000',
        'AGENT_BATCH_SIZE': '10',
        'INTERACTION_INTERVAL': '1',
        'COMMUNITY_SIZE_LIMIT': '100',
        'DATABASE_URL': 'sqlite:///:memory:',
        'LOG_LEVEL': 'DEBUG'
    }

    os.environ.update(test_env)
    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def missing_env_vars():
    """Test environment with missing required variables"""
    original_env = os.environ.copy()

    # Clear environment variables
    required_vars = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL']
    for var in required_vars:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)