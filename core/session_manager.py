"""Session management for Ollama Chat Streamer.

This module provides functionality for listing, selecting, and exporting saved
conversation sessions from the database.
"""

import argparse
import csv
import json
import sys
import textwrap
from typing import List, Dict, Any, Optional, TextIO
from pathlib import Path

# Optional interactive menu library
try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

# Database integration (optional)
from typing import List, Optional, Dict, Any, TextIO
from pathlib import Path
import sys
import json
import csv
import textwrap

try:
    from db import get_database_manager, Conversation
    HAS_DB = True
except ImportError:
    HAS_DB = False


async def list_sessions() -> None:
    """List all saved conversation sessions and exit."""
    if not HAS_DB:
        print("Error: Database dependencies not installed. Run: pip install psycopg2-binary asyncpg")
        return

    dbm = get_database_manager(sync=True)
    await dbm.create_tables()
    sessions = await dbm.get_all_conversations()
    
    if not sessions:
        print("No saved sessions found.")
        return
    
    for conv in sessions:
        print(f"ID: {conv.id} | Model: {conv.model} | Created: {conv.created_at}")


async def select_session() -> Optional[List[Dict[str, str]]]:
    """Interactively select a saved session to continue.
    
    Returns:
        List of messages from the selected session, or None if no session was selected.
    """
    if not HAS_DB:
        print("Error: Database dependencies not installed. Run: pip install psycopg2-binary asyncpg")
        return None

    dbm = get_database_manager(sync=True)
    await dbm.create_tables()
    sessions = await dbm.get_all_conversations()
    
    if not sessions:
        print("No saved sessions found.")
        return None
    
    choices = [f"{c.id}: {c.model} ({c.created_at})" for c in sessions]
    
    if HAS_QUESTIONARY:
        answer = questionary.select("Select a conversation to continue:", choices=choices).ask()
    else:
        print("Select a conversation to continue:")
        for i, choice in enumerate(choices, 1):
            print(f"{i}) {choice}")
        sel = input("Enter number: ")
        try:
            idx = int(sel) - 1
            answer = choices[idx]
        except Exception:
            print("Invalid selection.")
            return None
    
    selected_id = int(answer.split(":")[0])
    conv = await dbm.get_conversation(selected_id)
    
    if conv is None:
        print(f"Conversation {selected_id} not found.")
        return None
    
    print(f"Resuming conversation ID {selected_id} (model={conv.model})")
    return conv.messages


async def export_session(session_id: int, output_format: str = "json", output_target: Optional[TextIO] = None) -> None:
    """Export a conversation session in the specified format.
    
    Args:
        session_id: ID of the session to export.
        output_format: Format to export (json, csv, text, sql).
        output_target: File handle to write to. If None, prints to stdout.
    """
    if not HAS_DB:
        print("Error: Database dependencies not installed. Run: pip install psycopg2-binary asyncpg")
        return

    dbm = get_database_manager(sync=True)
    await dbm.create_tables()
    conv = await dbm.get_conversation(session_id)
    
    if conv is None:
        print(f"Conversation {session_id} not found.")
        return
    
    out_target = sys.stdout if output_target is None else output_target
    
    try:
        if output_format == "json":
            json.dump({
                "id": conv.id,
                "model": conv.model,
                "flags": conv.flags,
                "messages": conv.messages,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            }, out_target, indent=2)
        
        elif output_format == "csv":
            writer = csv.writer(out_target)
            writer.writerow(["role", "content"])
            for msg in conv.messages:
                writer.writerow([msg.get("role", ""), msg.get("content", "")])
        
        elif output_format == "text":
            for msg in conv.messages:
                out_target.write(f"{msg.get('role', '')}: {msg.get('content', '')}\n\n")
        
        elif output_format == "sql":
            sql = textwrap.dedent(f"""
                INSERT INTO conversations (id, model, flags, messages, created_at, updated_at)
                VALUES ({conv.id}, '{conv.model}', '{json.dumps(conv.flags)}', '{json.dumps(conv.messages)}',
                '{conv.created_at.isoformat() if conv.created_at else None}',
                '{conv.updated_at.isoformat() if conv.updated_at else None}');
            """)
            out_target.write(sql)
        
        print(f"Conversation {session_id} exported as {output_format}.")
    
    finally:
        if output_target is not None and output_target is not sys.stdout:
            out_target.close()