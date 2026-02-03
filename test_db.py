#!/usr/bin/env python3
"""
Test script for database functionality.
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from db import get_database_manager, Conversation
    HAS_DB = True
except ImportError:
    HAS_DB = False
    print("Error: Database dependencies not installed.")
    print("Run: pip install psycopg2-binary asyncpg")
    sys.exit(1)


def test_database():
    """Test basic database operations."""
    print("Testing database functionality...")
    
    try:
        # Get database manager
        db_manager = get_database_manager(sync=True)
        print("✓ Database manager created")
        
        # Create tables
        db_manager.create_tables()
        print("✓ Tables created/verified")
        
        # Test saving a conversation
        test_messages = [
            {"role": "user", "content": "Hello, this is a test message."},
            {"role": "assistant", "content": "Hello! I'm here to help."},
            {"role": "user", "content": "Can you help me test the database?"},
            {"role": "assistant", "content": "Of course! The database integration is working correctly."}
        ]
        
        test_flags = {
            "experimental": True,
            "web_search": False,
            "test_run": True
        }
        
        conversation_id = db_manager.save_conversation(
            model="test-model",
            messages=test_messages,
            flags=test_flags
        )
        print(f"✓ Saved test conversation with ID: {conversation_id}")
        
        # Test retrieving the conversation
        retrieved = db_manager.get_conversation(conversation_id)
        if retrieved:
            print(f"✓ Retrieved conversation {conversation_id}")
            print(f"  Model: {retrieved.model}")
            print(f"  Messages: {len(retrieved.messages)}")
            print(f"  Created: {retrieved.created_at}")
        else:
            print("✗ Failed to retrieve conversation")
            return False
        
        # Test updating the conversation
        updated_messages = test_messages + [
            {"role": "user", "content": "This is an additional test message."},
            {"role": "assistant", "content": "The update functionality is working!"}
        ]
        
        success = db_manager.update_conversation(conversation_id, updated_messages)
        if success:
            print("✓ Conversation updated successfully")
            
            # Verify the update
            updated = db_manager.get_conversation(conversation_id)
            if updated and len(updated.messages) == len(updated_messages):
                print("✓ Update verified - message count increased")
            else:
                print("✗ Update verification failed")
        else:
            print("✗ Failed to update conversation")
            return False
        
        # Test getting all conversations
        all_conversations = db_manager.get_all_conversations()
        print(f"✓ Retrieved {len(all_conversations)} total conversations")
        
        # Test filtering by model
        model_conversations = db_manager.get_all_conversations(model="test-model")
        print(f"✓ Found {len(model_conversations)} conversations for test-model")
        
        # Test limit
        limited_conversations = db_manager.get_all_conversations(limit=5)
        print(f"✓ Retrieved {len(limited_conversations)} conversations with limit=5")
        
        # Test deletion
        deleted = db_manager.delete_conversation(conversation_id)
        if deleted:
            print("✓ Conversation deleted successfully")
            
            # Verify deletion
            deleted_conv = db_manager.get_conversation(conversation_id)
            if deleted_conv is None:
                print("✓ Deletion verified - conversation no longer exists")
            else:
                print("✗ Deletion verification failed")
        else:
            print("✗ Failed to delete conversation")
            return False
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_loading():
    """Test the context loading functionality."""
    print("\nTesting context loading...")
    
    try:
        from stream_chat import load_context_from_database, parse_context_arg
        
        # Test parsing context arguments
        test_cases = [
            ("db", {"use_db": True, "paths": []}),
            ("db,.", {"use_db": True, "paths": ["."]}),
            ("db,./notes,./docs", {"use_db": True, "paths": ["./notes", "./docs"]}),
            ("./files", {"use_db": False, "paths": ["./files"]}),
            ("", {"use_db": False, "paths": []}),
        ]
        
        for input_val, expected in test_cases:
            result = parse_context_arg(input_val)
            if result == expected:
                print(f"✓ Parsed '{input_val}' correctly: {result}")
            else:
                print(f"✗ Failed to parse '{input_val}': got {result}, expected {expected}")
                return False
        
        print("✓ Context argument parsing tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Context loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Database Integration Test Suite")
    print("=" * 40)
    
    # Check if database URL is set
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Warning: DATABASE_URL environment variable not set.")
        print("Please set it in your .env file or environment.")
        print("\nExample:")
        print("export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/chatdb'")
        print("\nFor Docker Compose:")
        print("export DATABASE_URL='postgresql://postgres:postgres@postgres:5432/chatdb'")
        print("\nContinuing with test anyway...")
    
    # Run tests
    db_success = test_database()
    context_success = test_context_loading()
    
    print("\n" + "=" * 40)
    if db_success and context_success:
        print("🎉 All tests passed! Database integration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()