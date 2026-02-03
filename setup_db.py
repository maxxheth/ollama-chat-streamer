#!/usr/bin/env python3
"""
Setup script for initializing the PostgreSQL database for the chat app.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from db import get_database_manager
    HAS_DB = True
except ImportError:
    HAS_DB = False
    print("Error: Database dependencies not installed.")
    print("Run: pip install psycopg2-binary asyncpg")
    sys.exit(1)


def setup_database():
    """Initialize the database tables."""
    try:
        print("Setting up database...")
        
        # Get database manager
        db_manager = get_database_manager(sync=True)
        
        # Create tables
        db_manager.create_tables()
        print("✓ Database tables created successfully!")
        
        # Test connection
        conversations = db_manager.get_all_conversations(limit=1)
        print(f"✓ Database connection verified. Found {len(conversations)} existing conversations.")
        
        print("\nDatabase setup complete!")
        print("\nTo use the database features:")
        print("1. Set DATABASE_URL environment variable or add it to .env file")
        print("2. Run the chat app with --persist-to-db flag")
        print("3. Use --context db to load past conversations")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running")
        print("2. Check DATABASE_URL format: postgresql://user:password@host:port/database")
        print("3. Verify database credentials and permissions")
        sys.exit(1)


def show_database_info():
    """Show information about the database and existing conversations."""
    try:
        db_manager = get_database_manager(sync=True)
        
        # Get all conversations
        conversations = db_manager.get_all_conversations()
        
        print(f"Database: {db_manager.database_url}")
        print(f"Total conversations: {len(conversations)}")
        
        if conversations:
            print("\nRecent conversations:")
            for conv in conversations[:5]:  # Show last 5
                print(f"  ID: {conv.id}, Model: {conv.model}, "
                      f"Messages: {len(conv.messages)}, "
                      f"Created: {conv.created_at}")
        
    except Exception as e:
        print(f"Error accessing database: {e}")


def main():
    """Main setup function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "info":
            show_database_info()
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Database setup script for ollama-chat-streamer")
            print("\nUsage:")
            print("  python setup_db.py          # Initialize database tables")
            print("  python setup_db.py info     # Show database information")
            print("  python setup_db.py --help   # Show this help")
            return
    
    setup_database()


if __name__ == "__main__":
    main()