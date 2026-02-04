"""Context loading utilities for Ollama Chat Streamer.

This module provides functionality for loading context from files, directories,
and the database to inject into the LLM's prompt.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Database integration (optional)
try:
    from db import get_database_manager, Conversation
    HAS_DB = True
except ImportError:
    HAS_DB = False


def parse_context_arg(context_arg: str) -> Dict[str, Any]:
    """Parse the --context argument to extract database and file paths.
    
    Args:
        context_arg: The context argument value (e.g., "db,./docs").
        
    Returns:
        Dictionary with 'use_db' boolean and 'paths' list of additional paths.
    """
    if not context_arg:
        return {'use_db': False, 'paths': []}
    
    # Split by comma to handle multiple sources
    parts = [part.strip() for part in context_arg.split(',')]
    
    use_db = False
    paths = []
    
    for part in parts:
        if part.lower() == 'db':
            use_db = True
        elif part:
            paths.append(part)
    
    return {'use_db': use_db, 'paths': paths}


def load_context_files(context_path: str, extensions: List[str]) -> str:
    """Load text content from a file or recursively from a directory.
    
    Args:
        context_path: Path to a file or directory.
        extensions: List of file extensions to include (e.g., ["txt", "md"]).
        
    Returns:
        Concatenated content of all matching files.
    """
    if not context_path or not os.path.exists(context_path):
        return ""
    
    context_parts = []
    
    # Normalize extensions (remove dots if present)
    extensions = [ext.lstrip('.').lower() for ext in extensions]
    
    if os.path.isfile(context_path):
        # Single file
        try:
            with open(context_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                context_parts.append(f"=== File: {context_path} ===\n{content}\n")
        except Exception as e:
            context_parts.append(f"=== Error reading {context_path}: {e} ===\n")
    
    elif os.path.isdir(context_path):
        # Directory - recursively find files with matching extensions
        for root, _, files in os.walk(context_path):
            for file in files:
                ext = file.split('.')[-1].lower() if '.' in file else ''
                if ext in extensions or '*' in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            rel_path = os.path.relpath(file_path, context_path)
                            context_parts.append(f"=== File: {rel_path} ===\n{content}\n")
                    except Exception as e:
                        context_parts.append(f"=== Error reading {file_path}: {e} ===\n")
    
    return "\n".join(context_parts)


def load_context_from_database(
    db_manager: Any, 
    additional_paths: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> str:
    """Load context from database and optionally from additional file paths.
    
    Args:
        db_manager: Database manager instance.
        additional_paths: Additional file/directory paths to load context from.
        limit: Maximum number of conversations to load from database.
        
    Returns:
        Concatenated context from database and files.
    """
    context_parts = []
    
    # Load from database
    try:
        conversations = db_manager.get_all_conversations(limit=limit)
        if conversations:
            context_parts.append("=== Database Conversations ===\n")
            for conv in conversations:
                context_parts.append(f"--- Conversation {conv.id} (Model: {conv.model}, Created: {conv.created_at}) ---\n")
                for msg in conv.messages:
                    if msg.get('role') in ['user', 'assistant']:
                        role = msg.get('role', 'unknown').upper()
                        content = msg.get('content', '')
                        context_parts.append(f"{role}: {content}\n")
                context_parts.append("\n")
    except Exception as e:
        context_parts.append(f"=== Error loading from database: {e} ===\n")
    
    # Load from additional paths
    if additional_paths:
        for path in additional_paths:
            if path and path != 'db':
                content = load_context_files(path, ['txt', 'log'])
                if content:
                    context_parts.append(f"=== File Context: {path} ===\n{content}\n")
    
    return "\n".join(context_parts)