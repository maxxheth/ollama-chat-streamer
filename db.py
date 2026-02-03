"""
Database module for storing and retrieving chat conversations in PostgreSQL.
"""
import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


@dataclass
class Conversation:
    """Represents a chat conversation."""
    id: Optional[int] = None
    model: str = ""
    flags: Dict[str, Any] = None
    messages: List[Dict[str, str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = {}
        if self.messages is None:
            self.messages = []


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            database_url: PostgreSQL connection URL. If None, uses DATABASE_URL env var.
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "Database URL must be provided or set in DATABASE_URL environment variable"
            )
    
    async def create_tables(self) -> None:
        """Create the conversations table if it doesn't exist."""
        conn = await asyncpg.connect(self.database_url)
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    model VARCHAR(255) NOT NULL,
                    flags JSONB DEFAULT '{}',
                    messages JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create indexes for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_created_at 
                ON conversations(created_at DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_model 
                ON conversations(model)
            """)
        finally:
            await conn.close()
    
    async def save_conversation(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        flags: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a conversation to the database.
        
        Args:
            model: The model used for the conversation
            messages: List of message dictionaries
            flags: Additional flags/metadata
            
        Returns:
            The ID of the saved conversation
        """
        conn = await asyncpg.connect(self.database_url)
        try:
            flags = flags or {}
            result = await conn.fetchval("""
                INSERT INTO conversations (model, flags, messages)
                VALUES ($1, $2, $3)
                RETURNING id
            """, model, json.dumps(flags), json.dumps(messages))
            return result
        finally:
            await conn.close()
    
    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            Conversation object or None if not found
        """
        conn = await asyncpg.connect(self.database_url)
        try:
            row = await conn.fetchrow("""
                SELECT id, model, flags, messages, created_at, updated_at
                FROM conversations
                WHERE id = $1
            """, conversation_id)
            
            if row is None:
                return None
            
            return Conversation(
                id=row['id'],
                model=row['model'],
                flags=row['flags'] or {},
                messages=row['messages'] or [],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        finally:
            await conn.close()
    
    async def get_all_conversations(
        self, 
        limit: Optional[int] = None,
        model: Optional[str] = None
    ) -> List[Conversation]:
        """
        Retrieve all conversations, optionally filtered by model.
        
        Args:
            limit: Maximum number of conversations to return
            model: Optional model name filter
            
        Returns:
            List of Conversation objects
        """
        conn = await asyncpg.connect(self.database_url)
        try:
            query = """
                SELECT id, model, flags, messages, created_at, updated_at
                FROM conversations
            """
            params = []
            
            if model:
                query += " WHERE model = $1"
                params.append(model)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                if model:
                    query += f" LIMIT ${len(params) + 1}"
                else:
                    query += " LIMIT $1"
                params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            conversations = []
            for row in rows:
                conversations.append(Conversation(
                    id=row['id'],
                    model=row['model'],
                    flags=row['flags'] or {},
                    messages=row['messages'] or [],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))
            
            return conversations
        finally:
            await conn.close()
    
    async def update_conversation(
        self, 
        conversation_id: int, 
        messages: List[Dict[str, str]]
    ) -> bool:
        """
        Update an existing conversation with new messages.
        
        Args:
            conversation_id: The ID of the conversation to update
            messages: Updated list of messages
            
        Returns:
            True if updated successfully, False otherwise
        """
        conn = await asyncpg.connect(self.database_url)
        try:
            result = await conn.execute("""
                UPDATE conversations 
                SET messages = $1, updated_at = NOW()
                WHERE id = $2
            """, json.dumps(messages), conversation_id)
            
            return result != "UPDATE 0"
        finally:
            await conn.close()
    
    async def delete_conversation(self, conversation_id: int) -> bool:
        """
        Delete a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        conn = await asyncpg.connect(self.database_url)
        try:
            result = await conn.execute("""
                DELETE FROM conversations WHERE id = $1
            """, conversation_id)
            
            return result != "DELETE 0"
        finally:
            await conn.close()
    
    async def close(self) -> None:
        """Close any database connections (placeholder for asyncpg)."""
        # asyncpg connections are automatically closed when using context manager
        pass


# Synchronous wrapper for backwards compatibility
class SyncDatabaseManager:
    """Synchronous wrapper for database operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the synchronous database manager.
        
        Args:
            database_url: PostgreSQL connection URL. If None, uses DATABASE_URL env var.
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "Database URL must be provided or set in DATABASE_URL environment variable"
            )
    
    def create_tables(self) -> None:
        """Create the conversations table if it doesn't exist."""
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary is required for synchronous database operations")
        
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        model VARCHAR(255) NOT NULL,
                        flags JSONB DEFAULT '{}',
                        messages JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_created_at 
                    ON conversations(created_at DESC)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_model 
                    ON conversations(model)
                """)
            
            conn.commit()
        finally:
            conn.close()
    
    def save_conversation(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        flags: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a conversation to the database.
        
        Args:
            model: The model used for the conversation
            messages: List of message dictionaries
            flags: Additional flags/metadata
            
        Returns:
            The ID of the saved conversation
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary is required for synchronous database operations")
        
        conn = psycopg2.connect(self.database_url)
        try:
            flags = flags or {}
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations (model, flags, messages)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (model, json.dumps(flags), json.dumps(messages)))
                
                conversation_id = cur.fetchone()[0]
                conn.commit()
                return conversation_id
        finally:
            conn.close()
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            Conversation object or None if not found
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary is required for synchronous database operations")
        
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, model, flags, messages, created_at, updated_at
                    FROM conversations
                    WHERE id = %s
                """, (conversation_id,))
                
                row = cur.fetchone()
                if row is None:
                    return None
                
                return Conversation(
                    id=row['id'],
                    model=row['model'],
                    flags=row['flags'] or {},
                    messages=row['messages'] or [],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
        finally:
            conn.close()
    
    def get_all_conversations(
        self, 
        limit: Optional[int] = None,
        model: Optional[str] = None
    ) -> List[Conversation]:
        """
        Retrieve all conversations, optionally filtered by model.
        
        Args:
            limit: Maximum number of conversations to return
            model: Optional model name filter
            
        Returns:
            List of Conversation objects
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary is required for synchronous database operations")
        
        conn = psycopg2.connect(self.database_url)
        try:
            query = """
                SELECT id, model, flags, messages, created_at, updated_at
                FROM conversations
            """
            params = []
            
            if model:
                query += " WHERE model = %s"
                params.append(model)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                conversations = []
                for row in rows:
                    conversations.append(Conversation(
                        id=row['id'],
                        model=row['model'],
                        flags=row['flags'] or {},
                        messages=row['messages'] or [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    ))
                
                return conversations
        finally:
            conn.close()
    
    def update_conversation(
        self, 
        conversation_id: int, 
        messages: List[Dict[str, str]]
    ) -> bool:
        """
        Update an existing conversation with new messages.
        
        Args:
            conversation_id: The ID of the conversation to update
            messages: Updated list of messages
            
        Returns:
            True if updated successfully, False otherwise
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary is required for synchronous database operations")
        
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE conversations 
                    SET messages = %s, updated_at = NOW()
                    WHERE id = %s
                """, (json.dumps(messages), conversation_id))
                
                updated = cur.rowcount > 0
                conn.commit()
                return updated
        finally:
            conn.close()
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Delete a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2-binary is required for synchronous database operations")
        
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM conversations WHERE id = %s
                """, (conversation_id,))
                
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        finally:
            conn.close()
    
    def close(self) -> None:
        """Close any database connections (placeholder for psycopg2)."""
        # psycopg2 connections are automatically closed when using context manager
        pass


def get_database_manager(sync: bool = False) -> DatabaseManager:
    """
    Get a database manager instance.
    
    Args:
        sync: If True, return synchronous manager; if False, return async manager
        
    Returns:
        DatabaseManager instance
    """
    if sync:
        return SyncDatabaseManager()
    else:
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg is required for async database operations")
        return DatabaseManager()