"""
Session Archive module for the AI Trading Agent.

This module provides functionality to archive and retrieve paper trading sessions.
"""

from typing import Dict, Any, List, Optional, Union
import json
import os
import shutil
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
import sqlite3

from ..common import logger
from .session_manager import PaperTradingSession, session_manager


class SessionArchive:
    """
    Provides functionality to archive and retrieve paper trading sessions.
    """
    
    def __init__(self, archive_dir: str = None):
        """
        Initialize the session archive.
        
        Args:
            archive_dir: Directory for storing archived sessions
        """
        if archive_dir is None:
            # Default to an archives directory in the project
            archive_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data",
                "archives"
            )
        
        self.archive_dir = Path(archive_dir)
        
        # Create archives directory if it doesn't exist
        os.makedirs(self.archive_dir, exist_ok=True)
    
    def archive_session(self, session_id: str) -> Dict[str, Any]:
        """
        Archive a session.
        
        Args:
            session_id: ID of the session to archive
            
        Returns:
            Dictionary with archive information or error
        """
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}
        
        # Create a unique archive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filename = f"{session_id}_{timestamp}.zip"
        archive_path = self.archive_dir / archive_filename
        
        try:
            # Create a temporary directory for the archive contents
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create session data file
                session_data = session.to_db_dict()
                session_file = temp_path / "session.json"
                with open(session_file, "w") as f:
                    json.dump(session_data, f, indent=2)
                
                # Create results file
                results_file = temp_path / "results.json"
                with open(results_file, "w") as f:
                    json.dump(session.results, f, indent=2)
                
                # Create metadata file
                metadata = {
                    "archived_at": datetime.now().isoformat(),
                    "session_id": session_id,
                    "status": session.status,
                    "start_time": session.start_time,
                    "end_time": session.end_time,
                    "duration_minutes": session.duration_minutes,
                    "symbols": session.symbols,
                    "user_id": session.user_id
                }
                metadata_file = temp_path / "metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Create the zip archive
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(session_file, arcname="session.json")
                    zipf.write(results_file, arcname="results.json")
                    zipf.write(metadata_file, arcname="metadata.json")
            
            logger.info(f"Archived session {session_id} to {archive_path}")
            
            return {
                "success": True,
                "archive_path": str(archive_path),
                "archive_filename": archive_filename,
                "archived_at": metadata["archived_at"]
            }
        
        except Exception as e:
            logger.error(f"Error archiving session {session_id}: {str(e)}")
            return {"error": f"Error archiving session: {str(e)}"}
    
    def list_archives(self, user_id: str = None) -> List[Dict[str, Any]]:
        """
        List all archived sessions.
        
        Args:
            user_id: Optional user ID to filter archives
            
        Returns:
            List of archive metadata
        """
        archives = []
        
        try:
            for file_path in self.archive_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(file_path, "r") as zipf:
                        if "metadata.json" in zipf.namelist():
                            with zipf.open("metadata.json") as f:
                                metadata = json.load(f)
                                
                                # Filter by user_id if specified
                                if user_id and metadata.get("user_id") != user_id:
                                    continue
                                
                                archives.append({
                                    "archive_filename": file_path.name,
                                    "archive_path": str(file_path),
                                    "session_id": metadata.get("session_id"),
                                    "archived_at": metadata.get("archived_at"),
                                    "start_time": metadata.get("start_time"),
                                    "end_time": metadata.get("end_time"),
                                    "duration_minutes": metadata.get("duration_minutes"),
                                    "symbols": metadata.get("symbols"),
                                    "user_id": metadata.get("user_id")
                                })
                except Exception as e:
                    logger.error(f"Error reading archive {file_path}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error listing archives: {str(e)}")
        
        # Sort by archived_at timestamp (newest first)
        archives.sort(key=lambda x: x.get("archived_at", ""), reverse=True)
        
        return archives
    
    def get_archive_metadata(self, archive_filename: str) -> Dict[str, Any]:
        """
        Get metadata for an archived session.
        
        Args:
            archive_filename: Filename of the archive
            
        Returns:
            Archive metadata or error
        """
        archive_path = self.archive_dir / archive_filename
        
        if not archive_path.exists():
            return {"error": f"Archive {archive_filename} not found"}
        
        try:
            with zipfile.ZipFile(archive_path, "r") as zipf:
                if "metadata.json" in zipf.namelist():
                    with zipf.open("metadata.json") as f:
                        metadata = json.load(f)
                        metadata["archive_filename"] = archive_filename
                        metadata["archive_path"] = str(archive_path)
                        return metadata
                else:
                    return {"error": "Metadata not found in archive"}
        
        except Exception as e:
            logger.error(f"Error reading archive metadata {archive_filename}: {str(e)}")
            return {"error": f"Error reading archive metadata: {str(e)}"}
    
    def extract_archive(self, archive_filename: str, extract_dir: str = None) -> Dict[str, Any]:
        """
        Extract an archived session.
        
        Args:
            archive_filename: Filename of the archive
            extract_dir: Directory to extract to (optional)
            
        Returns:
            Dictionary with extraction information or error
        """
        archive_path = self.archive_dir / archive_filename
        
        if not archive_path.exists():
            return {"error": f"Archive {archive_filename} not found"}
        
        if extract_dir is None:
            # Create a temporary directory
            extract_dir = tempfile.mkdtemp()
        
        try:
            with zipfile.ZipFile(archive_path, "r") as zipf:
                zipf.extractall(extract_dir)
            
            logger.info(f"Extracted archive {archive_filename} to {extract_dir}")
            
            return {
                "success": True,
                "extract_dir": extract_dir
            }
        
        except Exception as e:
            logger.error(f"Error extracting archive {archive_filename}: {str(e)}")
            return {"error": f"Error extracting archive: {str(e)}"}
    
    def restore_session(self, archive_filename: str) -> Dict[str, Any]:
        """
        Restore a session from an archive.
        
        Args:
            archive_filename: Filename of the archive
            
        Returns:
            Dictionary with restoration information or error
        """
        # Extract the archive
        extract_result = self.extract_archive(archive_filename)
        if "error" in extract_result:
            return extract_result
        
        extract_dir = extract_result["extract_dir"]
        
        try:
            # Load session data
            session_file = os.path.join(extract_dir, "session.json")
            with open(session_file, "r") as f:
                session_data = json.load(f)
            
            # Load results
            results_file = os.path.join(extract_dir, "results.json")
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Create a new session with a new ID
            original_session_id = session_data["session_id"]
            new_session_id = f"{original_session_id}_restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create the session
            new_session = PaperTradingSession(
                session_id=new_session_id,
                config_path=session_data["config_path"],
                duration_minutes=session_data["duration_minutes"],
                interval_minutes=session_data["interval_minutes"],
                symbols=json.loads(session_data["symbols"]) if isinstance(session_data["symbols"], str) else session_data["symbols"],
                initial_capital=session_data.get("initial_capital", 10000.0),
                user_id=session_data.get("user_id")
            )
            
            # Set session properties
            new_session.status = "restored"
            new_session.start_time = session_data["start_time"]
            new_session.end_time = session_data["end_time"]
            new_session.uptime_seconds = session_data["uptime_seconds"]
            
            if session_data["current_portfolio"] and isinstance(session_data["current_portfolio"], str):
                new_session.current_portfolio = json.loads(session_data["current_portfolio"])
            else:
                new_session.current_portfolio = session_data["current_portfolio"]
            
            # Set results
            new_session.results = results
            
            # Add to session manager
            session_manager.add_session(new_session)
            
            # Clean up temporary directory
            shutil.rmtree(extract_dir)
            
            logger.info(f"Restored session {original_session_id} as {new_session_id}")
            
            return {
                "success": True,
                "original_session_id": original_session_id,
                "new_session_id": new_session_id,
                "status": "restored"
            }
        
        except Exception as e:
            logger.error(f"Error restoring session from {archive_filename}: {str(e)}")
            # Clean up temporary directory
            shutil.rmtree(extract_dir)
            return {"error": f"Error restoring session: {str(e)}"}
    
    def delete_archive(self, archive_filename: str) -> Dict[str, Any]:
        """
        Delete an archived session.
        
        Args:
            archive_filename: Filename of the archive
            
        Returns:
            Dictionary with deletion information or error
        """
        archive_path = self.archive_dir / archive_filename
        
        if not archive_path.exists():
            return {"error": f"Archive {archive_filename} not found"}
        
        try:
            os.remove(archive_path)
            logger.info(f"Deleted archive {archive_filename}")
            
            return {
                "success": True,
                "archive_filename": archive_filename
            }
        
        except Exception as e:
            logger.error(f"Error deleting archive {archive_filename}: {str(e)}")
            return {"error": f"Error deleting archive: {str(e)}"}


# Create singleton instance
session_archive = SessionArchive()
