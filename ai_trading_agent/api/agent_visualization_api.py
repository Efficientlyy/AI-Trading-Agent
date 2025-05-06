"""
Agent Visualization API endpoints.

This module provides API endpoints for tracking and visualizing agent activities.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..common import logger

# Create router
router = APIRouter(
    prefix="/api/agent-visualization",
    tags=["agent-visualization"],
    responses={404: {"description": "Not found"}},
)

# Event storage
_agent_events = []
_agent_states = {}
_component_states = {}
_max_events = 1000  # Maximum number of events to store


class AgentEvent(BaseModel):
    """Agent event model."""
    timestamp: float
    event_type: str
    component: str
    action: str
    data: Dict[str, Any]
    symbol: Optional[str] = None


class AgentState(BaseModel):
    """Agent state model."""
    component: str
    state: Dict[str, Any]
    last_updated: float


def record_agent_event(component: str, action: str, data: Dict[str, Any], symbol: Optional[str] = None) -> None:
    """
    Record an agent event.
    
    Args:
        component: Component that generated the event
        action: Action that was performed
        data: Event data
        symbol: Trading symbol (if applicable)
    """
    global _agent_events
    
    # Create event
    event = {
        "timestamp": time.time(),
        "event_type": "agent_action",
        "component": component,
        "action": action,
        "data": data,
        "symbol": symbol
    }
    
    # Add to events list
    _agent_events.append(event)
    
    # Trim events list if too long
    if len(_agent_events) > _max_events:
        _agent_events = _agent_events[-_max_events:]


def update_agent_state(component: str, state: Dict[str, Any]) -> None:
    """
    Update the state of an agent component.
    
    Args:
        component: Component name
        state: Component state
    """
    global _agent_states
    
    # Update state
    _agent_states[component] = {
        "component": component,
        "state": state,
        "last_updated": time.time()
    }


def update_component_state(component: str, state: Dict[str, Any]) -> None:
    """
    Update the state of a system component.
    
    Args:
        component: Component name
        state: Component state
    """
    global _component_states
    
    # Update state
    _component_states[component] = {
        "component": component,
        "state": state,
        "last_updated": time.time()
    }


@router.get("/events")
async def get_agent_events(limit: int = 100, component: Optional[str] = None, action: Optional[str] = None, 
                          symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get agent events with optional filtering.
    
    Args:
        limit: Maximum number of events to return
        component: Filter by component
        action: Filter by action
        symbol: Filter by symbol
    
    Returns:
        List of agent events
    """
    # Apply filters
    filtered_events = _agent_events
    
    if component:
        filtered_events = [e for e in filtered_events if e["component"] == component]
    
    if action:
        filtered_events = [e for e in filtered_events if e["action"] == action]
    
    if symbol:
        filtered_events = [e for e in filtered_events if e["symbol"] == symbol]
    
    # Sort by timestamp (newest first) and limit
    sorted_events = sorted(filtered_events, key=lambda e: e["timestamp"], reverse=True)
    limited_events = sorted_events[:limit]
    
    return limited_events


@router.get("/states")
async def get_agent_states() -> Dict[str, AgentState]:
    """
    Get the current state of all agent components.
    
    Returns:
        Dictionary mapping component names to their states
    """
    return _agent_states


@router.get("/component-states")
async def get_component_states() -> Dict[str, AgentState]:
    """
    Get the current state of all system components.
    
    Returns:
        Dictionary mapping component names to their states
    """
    return _component_states


@router.get("/activity-timeline")
async def get_activity_timeline(minutes: int = 30) -> Dict[str, Any]:
    """
    Get a timeline of agent activities for visualization.
    
    Args:
        minutes: Number of minutes to include in the timeline
    
    Returns:
        Timeline data structure
    """
    # Calculate start time
    start_time = time.time() - (minutes * 60)
    
    # Filter events by time
    recent_events = [e for e in _agent_events if e["timestamp"] >= start_time]
    
    # Group events by component
    events_by_component = {}
    for event in recent_events:
        component = event["component"]
        if component not in events_by_component:
            events_by_component[component] = []
        events_by_component[component].append(event)
    
    # Create timeline structure
    timeline = {
        "start_time": start_time,
        "end_time": time.time(),
        "components": list(events_by_component.keys()),
        "events_by_component": events_by_component,
        "event_count": len(recent_events)
    }
    
    return timeline


@router.get("/data-flow")
async def get_data_flow() -> Dict[str, Any]:
    """
    Get data flow information for visualization.
    
    Returns:
        Data flow structure
    """
    # Create data flow structure based on recent events
    data_sources = set()
    data_sinks = set()
    data_flows = []
    
    # Look for data flow events in the last 5 minutes
    recent_time = time.time() - (5 * 60)
    recent_events = [e for e in _agent_events if e["timestamp"] >= recent_time]
    
    for event in recent_events:
        # Look for data transfer events
        if event["action"] in ["data_received", "data_sent", "data_processed"]:
            source = event["data"].get("source")
            destination = event["data"].get("destination")
            
            if source and destination:
                data_sources.add(source)
                data_sinks.add(destination)
                
                # Add flow if not already present
                flow = {
                    "source": source,
                    "destination": destination,
                    "data_type": event["data"].get("data_type", "unknown"),
                    "timestamp": event["timestamp"]
                }
                
                # Check if this flow is already recorded
                if not any(f["source"] == flow["source"] and 
                          f["destination"] == flow["destination"] and
                          f["data_type"] == flow["data_type"] for f in data_flows):
                    data_flows.append(flow)
    
    return {
        "data_sources": list(data_sources),
        "data_sinks": list(data_sinks),
        "data_flows": data_flows
    }


@router.post("/clear")
async def clear_agent_events():
    """
    Clear all agent events.
    
    Returns:
        Status message
    """
    global _agent_events
    _agent_events = []
    return {"status": "cleared", "message": "All agent events have been cleared"}
