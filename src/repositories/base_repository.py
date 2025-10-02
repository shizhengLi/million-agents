"""
Base Repository pattern implementation
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime

T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """Base repository interface for common CRUD operations"""

    def __init__(self, session: Session, model_class: type):
        self.session = session
        self.model_class = model_class

    def create(self, data: Dict[str, Any]) -> T:
        """Create a new record"""
        instance = self.model_class(**data)
        self.session.add(instance)
        self.session.commit()
        self.session.refresh(instance)
        return instance

    def get_by_id(self, record_id: int) -> Optional[T]:
        """Get record by ID"""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first()

    def get_all(self) -> List[T]:
        """Get all records"""
        return self.session.query(self.model_class).all()

    def update(self, record_id: int, data: Dict[str, Any]) -> Optional[T]:
        """Update a record"""
        instance = self.get_by_id(record_id)
        if not instance:
            return None

        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        self.session.commit()
        self.session.refresh(instance)
        return instance

    def delete(self, record_id: int) -> bool:
        """Delete a record"""
        instance = self.get_by_id(record_id)
        if not instance:
            return False

        self.session.delete(instance)
        self.session.commit()
        return True

    def count(self) -> int:
        """Count all records"""
        return self.session.query(self.model_class).count()

    def get_with_pagination(self, page: int = 1, per_page: int = 10,
                           order_by: str = "id", order_desc: bool = False) -> List[T]:
        """Get records with pagination"""
        query = self.session.query(self.model_class)

        # Apply ordering
        order_column = getattr(self.model_class, order_by, self.model_class.id)
        if order_desc:
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(asc(order_column))

        # Apply pagination
        offset = (page - 1) * per_page
        return query.offset(offset).limit(per_page).all()

    def bulk_create(self, data_list: List[Dict[str, Any]]) -> List[T]:
        """Bulk create records"""
        instances = [self.model_class(**data) for data in data_list]
        self.session.add_all(instances)
        self.session.commit()

        # Refresh all instances to get IDs
        for instance in instances:
            self.session.refresh(instance)

        return instances

    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """Bulk update records (requires id in each update dict)"""
        updated_count = 0
        for update_data in updates:
            if 'id' in update_data:
                record_id = update_data.pop('id')
                if self.update(record_id, update_data):
                    updated_count += 1
        return updated_count

    def bulk_delete(self, record_ids: List[int]) -> int:
        """Bulk delete records"""
        deleted_count = self.session.query(self.model_class).filter(
            self.model_class.id.in_(record_ids)
        ).delete(synchronize_session=False)
        self.session.commit()
        return deleted_count

    def exists(self, record_id: int) -> bool:
        """Check if record exists"""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first() is not None

    def get_created_after(self, cutoff_time: datetime) -> List[T]:
        """Get records created after cutoff time"""
        return self.session.query(self.model_class).filter(
            self.model_class.created_at >= cutoff_time
        ).all()

    def get_updated_after(self, cutoff_time: datetime) -> List[T]:
        """Get records updated after cutoff time"""
        return self.session.query(self.model_class).filter(
            self.model_class.updated_at >= cutoff_time
        ).all()

    def search(self, query: str, fields: List[str] = None, limit: int = 50) -> List[T]:
        """Generic search across specified fields"""
        if not fields:
            # Default to common text fields
            fields = ['name', 'description', 'title']

        conditions = []
        for field in fields:
            if hasattr(self.model_class, field):
                field_attr = getattr(self.model_class, field)
                conditions.append(field_attr.ilike(f"%{query}%"))

        if conditions:
            return self.session.query(self.model_class).filter(
                or_(*conditions)
            ).limit(limit).all()

        return []

    def filter_by(self, filters: Dict[str, Any]) -> List[T]:
        """Filter records by multiple criteria"""
        query = self.session.query(self.model_class)

        for key, value in filters.items():
            if hasattr(self.model_class, key):
                field_attr = getattr(self.model_class, key)

                if isinstance(value, (list, tuple)):
                    # Handle list/tuple values with IN clause
                    query = query.filter(field_attr.in_(value))
                elif isinstance(value, dict):
                    # Handle range queries
                    if 'min' in value:
                        query = query.filter(field_attr >= value['min'])
                    if 'max' in value:
                        query = query.filter(field_attr <= value['max'])
                else:
                    # Simple equality
                    query = query.filter(field_attr == value)

        return query.all()

    def get_random(self, limit: int = 1) -> List[T]:
        """Get random records"""
        import random
        all_records = self.get_all()
        if limit >= len(all_records):
            return all_records
        return random.sample(all_records, limit)

    def get_latest(self, limit: int = 10, field: str = "created_at") -> List[T]:
        """Get latest records by specified field"""
        if hasattr(self.model_class, field):
            order_column = getattr(self.model_class, field)
            return self.session.query(self.model_class).order_by(
                desc(order_column)
            ).limit(limit).all()
        return []

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate data against model constraints (basic implementation)"""
        errors = {}

        # Check for required fields based on model class
        if hasattr(self.model_class, '__tablename__'):
            table_name = self.model_class.__tablename__

            # Basic validation for common fields
            if 'name' in data and not data['name'].strip():
                errors['name'] = ['Name cannot be empty']

            if 'email' in data and '@' not in data['email']:
                errors['email'] = ['Invalid email format']

        return errors