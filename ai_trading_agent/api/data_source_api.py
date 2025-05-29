"""
Data Source API Module

This module provides API endpoints for managing the data source (mock/real)
in the trading system, allowing the UI to toggle between sources and 
get the current status.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..config.data_source_config import get_data_source_config
from ..data.data_source_factory import get_data_source_factory
from ..agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ..common.utils import get_logger

# Create router
router = APIRouter(
    prefix="/api/data-source",
    tags=["data-source"],
)

# Define response models
class DataSourceResponse(BaseModel):
    """Response model for data source status."""
    source_type: str
    use_mock_data: bool
    mock_data_settings: Dict[str, Any]
    real_data_settings: Dict[str, Any]


class ToggleResponse(BaseModel):
    """Response model for data source toggle."""
    previous: str
    current: str
    success: bool
    message: str


class MockSettingsUpdateRequest(BaseModel):
    """Request model for updating mock data settings."""
    volatility: Optional[float] = None
    trend_strength: Optional[float] = None
    seed: Optional[int] = None
    generate_regimes: Optional[bool] = None


# Create global logger
logger = get_logger("DataSourceAPI")

# Store reference to the TA agent singleton
_ta_agent = None


def get_ta_agent() -> AdvancedTechnicalAnalysisAgent:
    """
    Get or create the Technical Analysis Agent singleton.
    
    Returns:
        AdvancedTechnicalAnalysisAgent instance
    """
    global _ta_agent
    if _ta_agent is None:
        # Create with default configuration
        _ta_agent = AdvancedTechnicalAnalysisAgent()
        logger.info("Created Technical Analysis Agent for API access")
    return _ta_agent


@router.get("/status", response_model=DataSourceResponse)
async def get_data_source_status() -> DataSourceResponse:
    """
    Get the current data source status.
    
    Returns:
        DataSourceResponse with current status
    """
    config = get_data_source_config()
    use_mock = config.use_mock_data
    source_type = "mock" if use_mock else "real"
    
    return DataSourceResponse(
        source_type=source_type,
        use_mock_data=use_mock,
        mock_data_settings=config.get_mock_data_settings(),
        real_data_settings=config.get_real_data_settings()
    )


@router.post("/toggle", response_model=ToggleResponse)
async def toggle_data_source(agent: AdvancedTechnicalAnalysisAgent = Depends(get_ta_agent)) -> ToggleResponse:
    """
    Toggle between mock and real data sources.
    
    Returns:
        ToggleResponse with previous and current data source types
    """
    try:
        previous = agent.get_data_source_type()
        current = agent.toggle_data_source()
        
        logger.info(f"Data source toggled from {previous} to {current} via API")
        
        return ToggleResponse(
            previous=previous,
            current=current,
            success=True,
            message=f"Successfully toggled data source from {previous} to {current}"
        )
    except Exception as e:
        logger.error(f"Error toggling data source: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle data source: {str(e)}"
        )


@router.post("/mock-settings", response_model=DataSourceResponse)
async def update_mock_settings(
    settings: MockSettingsUpdateRequest,
    agent: AdvancedTechnicalAnalysisAgent = Depends(get_ta_agent)
) -> DataSourceResponse:
    """
    Update mock data generation settings.
    
    Args:
        settings: New settings for mock data generation
        
    Returns:
        DataSourceResponse with updated status
    """
    try:
        # Build update dict with only provided fields
        update_dict = {"mock_data_settings": {}}
        for field, value in settings.dict().items():
            if value is not None:
                update_dict["mock_data_settings"][field] = value
        
        # Update config if there are changes
        if update_dict["mock_data_settings"]:
            config = get_data_source_config()
            config.update_config(update_dict)
            logger.info(f"Updated mock data settings via API: {update_dict['mock_data_settings']}")
        
        # Return current status
        return await get_data_source_status()
    except Exception as e:
        logger.error(f"Error updating mock data settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update mock data settings: {str(e)}"
        )
