"""
Session Management API for the AI Trading Agent.

This module provides API endpoints for managing paper trading sessions,
including templates, comparison, and archiving.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
import asyncio
from datetime import datetime

from ..agent.session_manager import session_manager, PaperTradingSession
from ..agent.session_templates import template_manager, SessionTemplate
from ..agent.session_comparison import SessionComparison
from ..agent.session_archive import session_archive
from ..common import logger
from ..common.error_handling import TradingAgentError, ErrorCode, ErrorCategory, ErrorSeverity


router = APIRouter(prefix="/api/sessions", tags=["Session Management"])


# Session Templates API
@router.get("/templates")
async def get_templates(tag: Optional[str] = None):
    """
    Get all session templates or filter by tag.
    
    Args:
        tag: Optional tag to filter templates
    
    Returns:
        List of templates
    """
    try:
        if tag:
            templates = template_manager.get_templates_by_tag(tag)
        else:
            templates = template_manager.get_all_templates()
        
        return {
            "templates": [template.to_dict() for template in templates]
        }
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting templates: {str(e)}")


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """
    Get a specific template by ID.
    
    Args:
        template_id: ID of the template
    
    Returns:
        Template details
    """
    template = template_manager.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    
    return template.to_dict()


@router.post("/templates")
async def create_template(template_data: Dict[str, Any]):
    """
    Create a new session template.
    
    Args:
        template_data: Template data
    
    Returns:
        Created template
    """
    try:
        # Validate required fields
        required_fields = ["template_id", "name", "description", "config"]
        for field in required_fields:
            if field not in template_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Check if template ID already exists
        if template_manager.get_template(template_data["template_id"]):
            raise HTTPException(status_code=409, detail=f"Template ID {template_data['template_id']} already exists")
        
        # Create template
        template = SessionTemplate(
            template_id=template_data["template_id"],
            name=template_data["name"],
            description=template_data["description"],
            config=template_data["config"],
            tags=template_data.get("tags", [])
        )
        
        # Save template
        template_manager.save_template(template)
        
        return template.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating template: {str(e)}")


@router.put("/templates/{template_id}")
async def update_template(template_id: str, template_data: Dict[str, Any]):
    """
    Update an existing template.
    
    Args:
        template_id: ID of the template to update
        template_data: Updated template data
    
    Returns:
        Updated template
    """
    # Check if template exists
    existing_template = template_manager.get_template(template_id)
    if not existing_template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    
    try:
        # Update template
        template = SessionTemplate(
            template_id=template_id,
            name=template_data.get("name", existing_template.name),
            description=template_data.get("description", existing_template.description),
            config=template_data.get("config", existing_template.config),
            tags=template_data.get("tags", existing_template.tags)
        )
        
        # Save template
        template_manager.save_template(template)
        
        return template.to_dict()
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """
    Delete a template.
    
    Args:
        template_id: ID of the template to delete
    
    Returns:
        Deletion status
    """
    # Check if template exists
    if not template_manager.get_template(template_id):
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    
    # Delete template
    success = template_manager.delete_template(template_id)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Error deleting template {template_id}")
    
    return {"message": f"Template {template_id} deleted successfully"}


@router.post("/from-template/{template_id}")
async def create_session_from_template(
    template_id: str,
    override_config: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Create a new paper trading session from a template.
    
    Args:
        template_id: ID of the template to use
        override_config: Optional configuration overrides
        background_tasks: FastAPI background tasks
    
    Returns:
        Created session details
    """
    # Check if template exists
    template = template_manager.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    
    try:
        # Merge template config with overrides
        config = template.config.copy()
        if override_config:
            config.update(override_config)
        
        # Create session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{template_id}"
        
        # Create session
        session = PaperTradingSession(
            session_id=session_id,
            config_path=config.get("config_path", "config/trading_config.yaml"),
            duration_minutes=config.get("duration_minutes", 60),
            interval_minutes=config.get("interval_minutes", 1),
            symbols=config.get("symbols", ["BTC/USDT", "ETH/USDT"]),
            initial_capital=config.get("initial_capital", 10000.0),
            user_id=config.get("user_id")
        )
        
        # Add session to manager
        session_manager.add_session(session)
        
        # Start session in background if requested
        if config.get("auto_start", False) and background_tasks:
            background_tasks.add_task(session_manager.start_session, session_id)
        
        return {
            "session_id": session_id,
            "status": session.status,
            "message": f"Session created from template {template_id}",
            "template_name": template.name,
            "auto_start": config.get("auto_start", False)
        }
    except Exception as e:
        logger.error(f"Error creating session from template {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


# Session Comparison API
@router.post("/compare")
async def compare_sessions(session_ids: List[str]):
    """
    Compare multiple paper trading sessions.
    
    Args:
        session_ids: List of session IDs to compare
    
    Returns:
        Comparison report
    """
    if not session_ids or len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two session IDs are required for comparison")
    
    try:
        # Generate comparison report
        report = SessionComparison.generate_comparison_report(session_ids)
        
        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing sessions: {str(e)}")


@router.post("/compare/performance")
async def compare_performance(session_ids: List[str]):
    """
    Compare performance metrics across multiple sessions.
    
    Args:
        session_ids: List of session IDs to compare
    
    Returns:
        Performance comparison
    """
    if not session_ids or len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two session IDs are required for comparison")
    
    try:
        # Compare performance metrics
        comparison = SessionComparison.compare_performance_metrics(session_ids)
        
        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing performance metrics: {str(e)}")


@router.post("/compare/equity")
async def compare_equity_curves(session_ids: List[str]):
    """
    Compare equity curves across multiple sessions.
    
    Args:
        session_ids: List of session IDs to compare
    
    Returns:
        Equity curve comparison
    """
    if not session_ids or len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two session IDs are required for comparison")
    
    try:
        # Compare equity curves
        comparison = SessionComparison.compare_equity_curves(session_ids)
        
        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing equity curves: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing equity curves: {str(e)}")


@router.post("/compare/trades")
async def compare_trade_statistics(session_ids: List[str]):
    """
    Compare trade statistics across multiple sessions.
    
    Args:
        session_ids: List of session IDs to compare
    
    Returns:
        Trade statistics comparison
    """
    if not session_ids or len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two session IDs are required for comparison")
    
    try:
        # Compare trade statistics
        comparison = SessionComparison.compare_trade_statistics(session_ids)
        
        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing trade statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing trade statistics: {str(e)}")


# Session Archive API
@router.post("/archive/{session_id}")
async def archive_session(session_id: str):
    """
    Archive a paper trading session.
    
    Args:
        session_id: ID of the session to archive
    
    Returns:
        Archive information
    """
    try:
        # Archive session
        result = session_archive.archive_session(session_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error archiving session: {str(e)}")


@router.get("/archives")
async def list_archives(user_id: Optional[str] = None):
    """
    List all archived sessions.
    
    Args:
        user_id: Optional user ID to filter archives
    
    Returns:
        List of archives
    """
    try:
        # List archives
        archives = session_archive.list_archives(user_id)
        
        return {"archives": archives}
    except Exception as e:
        logger.error(f"Error listing archives: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing archives: {str(e)}")


@router.get("/archives/{archive_filename}")
async def get_archive_metadata(archive_filename: str):
    """
    Get metadata for an archived session.
    
    Args:
        archive_filename: Filename of the archive
    
    Returns:
        Archive metadata
    """
    try:
        # Get archive metadata
        metadata = session_archive.get_archive_metadata(archive_filename)
        
        if "error" in metadata:
            raise HTTPException(status_code=404, detail=metadata["error"])
        
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting archive metadata {archive_filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting archive metadata: {str(e)}")


@router.post("/archives/{archive_filename}/restore")
async def restore_session(archive_filename: str):
    """
    Restore a session from an archive.
    
    Args:
        archive_filename: Filename of the archive
    
    Returns:
        Restoration information
    """
    try:
        # Restore session
        result = session_archive.restore_session(archive_filename)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring session from {archive_filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error restoring session: {str(e)}")


@router.delete("/archives/{archive_filename}")
async def delete_archive(archive_filename: str):
    """
    Delete an archived session.
    
    Args:
        archive_filename: Filename of the archive
    
    Returns:
        Deletion information
    """
    try:
        # Delete archive
        result = session_archive.delete_archive(archive_filename)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting archive {archive_filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting archive: {str(e)}")


# Include the session management API in the main API router
def include_session_management_api(app):
    """
    Include the session management API in the main FastAPI app.
    
    Args:
        app: FastAPI application
    """
    app.include_router(router)
