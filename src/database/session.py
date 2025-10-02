"""
Database session management
"""

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine

from .config import create_engine as create_db_engine


# Global session factory
SessionLocal = None


def get_session() -> Session:
    """Get database session"""
    global SessionLocal
    if SessionLocal is None:
        engine = create_db_engine()
        SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_tables():
    """Create all database tables"""
    from .models import Base
    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)