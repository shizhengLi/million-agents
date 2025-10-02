"""
Database migration management using Alembic
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.runtime.environment import EnvironmentContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations using Alembic"""

    def __init__(self, database_url: str, script_location: str):
        self.database_url = database_url
        self.script_location = script_location
        self.config = None
        self.engine = None
        self._setup_config()

    def _setup_config(self):
        """Setup Alembic configuration"""
        self.config = Config("alembic.ini") if os.path.exists("alembic.ini") else Config()
        self.config.set_main_option("sqlalchemy.url", self.database_url)
        self.config.set_main_option("script_location", self.script_location)

        # Create script directory if it doesn't exist
        os.makedirs(self.script_location, exist_ok=True)
        versions_dir = os.path.join(self.script_location, "versions")
        os.makedirs(versions_dir, exist_ok=True)

        self.engine = create_engine(self.database_url)


def create_alembic_config(database_url: str, script_location: str) -> Config:
    """Create and configure Alembic configuration"""
    config = Config()
    config.set_main_option("sqlalchemy.url", database_url)
    config.set_main_option("script_location", script_location)
    config.set_main_option("file_template", "%%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s")
    return config


def setup_migration_environment(database_url: str, script_location: str) -> bool:
    """Setup complete migration environment"""
    try:
        # Create directories
        os.makedirs(script_location, exist_ok=True)
        versions_dir = os.path.join(script_location, "versions")
        os.makedirs(versions_dir, exist_ok=True)

        # Create alembic.ini if it doesn't exist
        alembic_ini_path = os.path.join(script_location, "alembic.ini")
        if not os.path.exists(alembic_ini_path):
            config = create_alembic_config(database_url, script_location)
            with open(alembic_ini_path, 'w') as f:
                f.write(f"""# Alembic configuration file
[alembic]
script_location = {script_location}
sqlalchemy.url = {database_url}

[post_write_hooks]

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
""")

        # Create env.py if it doesn't exist
        env_py_path = os.path.join(script_location, "env.py")
        if not os.path.exists(env_py_path):
            with open(env_py_path, 'w') as f:
                f.write('''"""Alembic environment configuration"""

import sys
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.models import Base
from src.database.config import get_database_url

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
''')

        # Create script.py.mako template
        script_template_path = os.path.join(script_location, "script.py.mako")
        if not os.path.exists(script_template_path):
            with open(script_template_path, 'w') as f:
                f.write('''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
''')

        logger.info(f"Migration environment setup complete at {script_location}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup migration environment: {e}")
        return False


def create_initial_migration(script_location: str, message: str = "Initial migration") -> Optional[str]:
    """Create initial migration"""
    try:
        # Create alembic.ini path
        alembic_ini_path = os.path.join(script_location, "alembic.ini")
        config = Config(alembic_ini_path)

        # Create initial migration
        command.revision(config, autogenerate=True, message=message)

        # Find the created migration file
        versions_dir = os.path.join(script_location, "versions")
        migration_files = [f for f in os.listdir(versions_dir) if f.endswith('.py')]

        if migration_files:
            latest_migration = sorted(migration_files)[-1]
            migration_path = os.path.join(versions_dir, latest_migration)
            logger.info(f"Initial migration created: {migration_path}")
            return migration_path

        return None

    except Exception as e:
        logger.error(f"Failed to create initial migration: {e}")
        return None


def run_upgrade(database_url: str, script_location: str, revision: str = "head", track_performance: bool = False) -> bool:
    """Run database upgrade migration"""
    try:
        start_time = time.time() if track_performance else None

        config = create_alembic_config(database_url, script_location)
        command.upgrade(config, revision)

        if track_performance and start_time:
            execution_time = time.time() - start_time
            logger.info(f"Migration upgrade completed in {execution_time:.2f} seconds")

        logger.info(f"Database upgraded to revision: {revision}")
        return True

    except Exception as e:
        logger.error(f"Failed to run upgrade migration: {e}")
        return False


def run_downgrade(database_url: str, script_location: str, revision: str = "base") -> bool:
    """Run database downgrade migration"""
    try:
        config = create_alembic_config(database_url, script_location)
        command.downgrade(config, revision)

        logger.info(f"Database downgraded to revision: {revision}")
        return True

    except Exception as e:
        logger.error(f"Failed to run downgrade migration: {e}")
        return False


def get_current_revision(database_url: str, script_location: str) -> Optional[str]:
    """Get current database revision"""
    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            return context.get_current_revision()

    except Exception as e:
        logger.error(f"Failed to get current revision: {e}")
        return None


def get_head_revision(script_location: str) -> Optional[str]:
    """Get head revision from migration scripts"""
    try:
        config = Config()
        config.set_main_option("script_location", script_location)
        script_dir = ScriptDirectory.from_config(config)
        return script_dir.get_current_head()

    except Exception as e:
        logger.error(f"Failed to get head revision: {e}")
        return None


def get_revision_history(script_location: str) -> List[str]:
    """Get complete revision history"""
    try:
        config = Config()
        config.set_main_option("script_location", script_location)
        script_dir = ScriptDirectory.from_config(config)

        revisions = []
        for revision in script_dir.walk_revisions():
            revisions.append(revision.revision)

        return list(reversed(revisions))

    except Exception as e:
        logger.error(f"Failed to get revision history: {e}")
        return []


def autogenerate_migration(database_url: str, script_location: str, message: str = "Auto-generated migration") -> Optional[str]:
    """Auto-generate migration from model changes"""
    try:
        # Use alembic.ini from script location
        alembic_ini_path = os.path.join(script_location, "alembic.ini")
        config = Config(alembic_ini_path)

        # Try to autogenerate
        command.revision(config, autogenerate=True, message=message)

        # Find the created migration file
        versions_dir = os.path.join(script_location, "versions")
        migration_files = [f for f in os.listdir(versions_dir) if f.endswith('.py')]

        if migration_files:
            latest_migration = sorted(migration_files)[-1]
            migration_path = os.path.join(versions_dir, latest_migration)
            logger.info(f"Auto-generated migration created: {migration_path}")
            return migration_path

        return None

    except Exception as e:
        logger.error(f"Failed to autogenerate migration: {e}")
        return None


def create_migration_branch(script_location: str, branch_name: str, message: str) -> bool:
    """Create a new migration branch"""
    try:
        # Use alembic.ini from script location
        alembic_ini_path = os.path.join(script_location, "alembic.ini")
        config = Config(alembic_ini_path)

        # Create branch migration
        command.revision(config, message=message, head=f"{branch_name}@head")

        logger.info(f"Migration branch '{branch_name}' created")
        return True

    except Exception as e:
        logger.error(f"Failed to create migration branch: {e}")
        return False


def rollback_migration(database_url: str, script_location: str, steps: int = 1) -> bool:
    """Rollback migration by specified number of steps"""
    try:
        current_rev = get_current_revision(database_url, script_location)
        if not current_rev:
            logger.warning("No current revision found")
            return False

        config = create_alembic_config(database_url, script_location)

        # Get revision history to find target revision
        history = get_revision_history(script_location)
        current_index = history.index(current_rev) if current_rev in history else -1

        if current_index < steps:
            target_rev = "base"
        else:
            target_rev = history[current_index - steps]

        command.downgrade(config, target_rev)
        logger.info(f"Rolled back {steps} step(s) to revision: {target_rev}")
        return True

    except Exception as e:
        logger.error(f"Failed to rollback migration: {e}")
        return False


def validate_migrations(script_location: str) -> bool:
    """Validate migration scripts for consistency and correctness"""
    try:
        config = Config()
        config.set_main_option("script_location", script_location)
        script_dir = ScriptDirectory.from_config(config)

        # Validate all revisions
        for revision in script_dir.walk_revisions():
            # Check if revision has proper upgrade/downgrade
            if not hasattr(revision.module, 'upgrade'):
                logger.error(f"Migration {revision.revision} missing upgrade function")
                return False
            if not hasattr(revision.module, 'downgrade'):
                logger.error(f"Migration {revision.revision} missing downgrade function")
                return False

        logger.info("Migration validation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Migration validation failed: {e}")
        return False


def check_migration_consistency(database_url: str, script_location: str) -> bool:
    """Check consistency between database state and migration files"""
    try:
        current_rev = get_current_revision(database_url, script_location)
        head_rev = get_head_revision(script_location)

        if current_rev and head_rev and current_rev != head_rev:
            logger.warning(f"Database revision {current_rev} differs from head revision {head_rev}")
            return False

        logger.info("Migration consistency check passed")
        return True

    except Exception as e:
        logger.error(f"Migration consistency check failed: {e}")
        return False


def backup_database_schema(database_url: str, backup_path: str) -> bool:
    """Backup current database schema"""
    try:
        engine = create_engine(database_url)

        with open(backup_path, 'w') as f:
            with engine.connect() as connection:
                # Get all table schemas
                result = connection.execute(text("SELECT sql FROM sqlite_master WHERE type='table'"))
                for row in result:
                    if row[0]:
                        f.write(f"{row[0]};\n")

        logger.info(f"Database schema backed up to: {backup_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to backup database schema: {e}")
        return False


def restore_database_schema(database_url: str, backup_path: str) -> bool:
    """Restore database schema from backup (SQLite specific)"""
    try:
        engine = create_engine(database_url)

        with open(backup_path, 'r') as f:
            schema_sql = f.read()

        with engine.connect() as connection:
            # SQLite specific restoration
            # Drop all existing tables first
            result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]

            # Drop tables except sqlite_sequence
            for table in tables:
                if table != 'sqlite_sequence':
                    connection.execute(text(f"DROP TABLE IF EXISTS {table}"))

            # Execute schema restoration
            statements = schema_sql.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    connection.execute(text(statement))

            connection.commit()

        logger.info(f"Database schema restored from: {backup_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to restore database schema: {e}")
        return False


def get_migration_performance_metrics(script_location: str) -> Dict[str, Any]:
    """Get migration performance metrics"""
    try:
        history = get_revision_history(script_location)

        return {
            'total_migrations': len(history),
            'script_location': script_location,
            'last_checked': datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get migration metrics: {e}")
        return {}


def get_migration_error_details(error: Exception) -> Dict[str, Any]:
    """Extract detailed error information from migration errors"""
    return {
        'error_type': type(error).__name__,
        'message': str(error),
        'timestamp': datetime.utcnow().isoformat(),
    }


def create_migration_config(database_url: str, script_location: str, environment: str = "development") -> Dict[str, Any]:
    """Create migration configuration"""
    return {
        'database_url': database_url,
        'script_location': script_location,
        'environment': environment,
        'created_at': datetime.utcnow().isoformat(),
    }


def save_migration_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save migration configuration to file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True

    except Exception as e:
        logger.error(f"Failed to save migration config: {e}")
        return False


def load_migration_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load migration configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Failed to load migration config: {e}")
        return None


def test_migration_reversibility(database_url: str, script_location: str) -> bool:
    """Test that all migrations can be reversed"""
    try:
        # Get current revision
        current_rev = get_current_revision(database_url, script_location)

        # Test downgrade
        if not run_downgrade(database_url, script_location, "base"):
            return False

        # Test upgrade back to original revision
        if not run_upgrade(database_url, script_location, current_rev or "head"):
            return False

        logger.info("Migration reversibility test passed")
        return True

    except Exception as e:
        logger.error(f"Migration reversibility test failed: {e}")
        return False


def test_migration_data_integrity(database_url: str, script_location: str) -> bool:
    """Test migration data integrity"""
    try:
        # This is a basic integrity test
        # In a real scenario, you would check specific data constraints
        engine = create_engine(database_url)

        with engine.connect() as connection:
            # Check if essential tables exist
            result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]

            # Basic integrity checks
            if 'alembic_version' not in tables:
                logger.warning("alembic_version table missing")
                return False

        logger.info("Migration data integrity test passed")
        return True

    except Exception as e:
        logger.error(f"Migration data integrity test failed: {e}")
        return False


def prepare_migration_deployment(script_location: str, environment: str) -> Optional[Dict[str, Any]]:
    """Prepare migration deployment package"""
    try:
        history = get_revision_history(script_location)

        return {
            'environment': environment,
            'migrations_to_run': len(history),
            'migration_list': history,
            'prepared_at': datetime.utcnow().isoformat(),
            'script_location': script_location,
        }

    except Exception as e:
        logger.error(f"Failed to prepare migration deployment: {e}")
        return None


def validate_deployment_readiness(database_url: str, script_location: str) -> bool:
    """Validate if environment is ready for migration deployment"""
    try:
        # Check database connectivity
        engine = create_engine(database_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        # Validate migration scripts
        if not validate_migrations(script_location):
            return False

        # Check consistency
        if not check_migration_consistency(database_url, script_location):
            return False

        logger.info("Deployment readiness validation passed")
        return True

    except Exception as e:
        logger.error(f"Deployment readiness validation failed: {e}")
        return False


def get_migration_logs(script_location: str) -> List[Dict[str, Any]]:
    """Get migration execution logs"""
    # This is a placeholder implementation
    # In a real scenario, you would read from actual log files
    return [
        {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO',
            'message': 'Migration logs not fully implemented',
        }
    ]


def monitor_migration_progress(database_url: str, script_location: str) -> Dict[str, Any]:
    """Monitor migration progress"""
    try:
        current_rev = get_current_revision(database_url, script_location)
        head_rev = get_head_revision(script_location)
        history = get_revision_history(script_location)

        progress_percentage = 0
        if current_rev and head_rev and current_rev != head_rev:
            try:
                current_index = history.index(current_rev)
                progress_percentage = (current_index / len(history)) * 100
            except ValueError:
                progress_percentage = 0
        elif current_rev == head_rev:
            progress_percentage = 100

        return {
            'current_revision': current_rev,
            'head_revision': head_rev,
            'progress_percentage': progress_percentage,
            'total_migrations': len(history),
            'status': 'completed' if progress_percentage == 100 else 'in_progress',
            'last_updated': datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to monitor migration progress: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'last_updated': datetime.utcnow().isoformat(),
        }