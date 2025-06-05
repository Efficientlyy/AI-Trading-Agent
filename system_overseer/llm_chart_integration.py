#!/usr/bin/env python
"""
LLM Integration for Chart Requests

This module integrates the natural language chart processor with the LLM and Telegram components.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple

# Import the natural language processor
import sys
sys.path.append('/home/ubuntu/projects/Trading-Agent')
from natural_language_chart_processor import NaturalLanguageChartProcessor

logger = logging.getLogger("system_overseer.llm_chart_integration")

class LLMChartIntegration:
    """Integrate LLM with chart generation for natural language requests."""
    
    def __init__(self):
        """Initialize the integration.
        
        Note: The core will be provided during the initialize method call
        """
        self.core = None
        self.nlp_processor = NaturalLanguageChartProcessor()
        self.llm_client = None
        self.telegram = None
        self.visualization = None
        
        logger.info("LLMChartIntegration instance created")
    
    def initialize(self, core) -> bool:
        """Initialize the integration with the system core.
        
        Args:
            core: System Overseer core instance
            
        Returns:
            bool: True if initialization was successful
        """
        self.core = core
        
        # Get required services
        self.llm_client = self.core.get_service("llm_client")
        if not self.llm_client:
            logger.error("LLM client service not found")
            return False
        
        self.telegram = self.core.get_service("telegram_integration")
        if not self.telegram:
            logger.error("Telegram integration service not found")
            return False
        
        # Get visualization plugin
        self.visualization = None
        plugin_manager = self.core.get_service("plugin_manager")
        if plugin_manager:
            self.visualization = plugin_manager.get_plugin("visualization")
            if not self.visualization:
                logger.error("Visualization plugin not found")
                return False
        else:
            logger.error("Plugin manager service not found")
            return False
        
        # Register message handler with Telegram
        self.telegram.register_message_handler(self.handle_message)
        
        logger.info("LLM chart integration initialized successfully")
        return True
    
    def start(self):
        """Start the plugin."""
        logger.info("LLM chart integration plugin started")
    
    def stop(self):
        """Stop the plugin."""
        logger.info("LLM chart integration plugin stopped")
    
    def handle_message(self, message: Dict[str, Any]) -> bool:
        """Handle incoming Telegram message.
        
        Args:
            message: Telegram message object
            
        Returns:
            bool: True if message was handled
        """
        # Skip command messages (they're handled by the command processor)
        if "text" not in message or message.get("text", "").startswith("/"):
            return False
        
        # Process message text
        text = message["text"]
        chat_id = message.get("chat", {}).get("id")
        
        if not chat_id:
            logger.error("Chat ID not found in message")
            return False
        
        # Check if this is a chart request
        is_chart_request, params = self.nlp_processor.process_request(text)
        
        if not is_chart_request:
            # Not a chart request, let other handlers process it
            return False
        
        # Log the extracted parameters
        logger.info(f"Chart request detected: {params}")
        
        # Generate chart
        self._generate_and_send_chart(chat_id, params)
        
        # Message was handled
        return True
    
    def _generate_and_send_chart(self, chat_id: int, params: Dict[str, Any]) -> None:
        """Generate chart and send it to the user.
        
        Args:
            chat_id: Telegram chat ID
            params: Chart parameters
        """
        symbol = params.get("symbol")
        chart_type = params.get("chart_type")
        interval = params.get("interval")
        indicators = params.get("indicators", [])
        
        if not symbol or not chart_type or not interval:
            logger.error(f"Missing required parameters: {params}")
            self.telegram.send_message(chat_id, "Sorry, I couldn't understand your chart request. Please try again with a clearer request.")
            return
        
        # Send acknowledgment message
        indicator_text = f" with {', '.join(indicators)}" if indicators else ""
        self.telegram.send_message(
            chat_id, 
            f"Generating {chart_type} chart for {symbol} ({interval}){indicator_text}..."
        )
        
        try:
            # Generate chart
            chart_data = self.visualization.get_chart(symbol, chart_type, interval, indicators)
            
            if not chart_data:
                logger.error(f"Failed to generate chart: {params}")
                self.telegram.send_message(
                    chat_id,
                    f"Sorry, I couldn't generate the {chart_type} chart for {symbol} ({interval}). Please try again later."
                )
                return
            
            # Save chart to temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(chart_data)
                temp_file_path = temp_file.name
            
            # Send chart to user
            self.telegram.send_photo(chat_id, temp_file_path)
            
            # Delete temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Failed to delete temporary file: {e}")
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            self.telegram.send_message(
                chat_id,
                f"Sorry, an error occurred while generating the chart: {str(e)}"
            )
