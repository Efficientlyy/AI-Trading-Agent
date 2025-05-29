import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketService, WebSocketTopic } from '../services/WebSocketService';

// Singleton instance of WebSocketService
const webSocketService = new WebSocketService();

// Types for sentiment pipeline data
export interface PipelineComponentData {
  name: string;
  status: 'online' | 'offline' | 'error' | 'processing';
  metrics: Record<string, any>;
  last_update: string;
  is_active: boolean;
}

export interface PipelineDataFlow {
  source: string;
  target: string;
  data_volume: number;
  timestamp: string;
  latency: number;
}

export interface SentimentPipelineSocketData {
  agent_id: string;
  pipeline_status: 'running' | 'stopped' | 'error';
  components: PipelineComponentData[];
  data_flows: PipelineDataFlow[];
  global_metrics: Record<string, any>;
  last_update: string;
  pipeline_latency: number;
}

/**
 * Custom hook for subscribing to real-time sentiment pipeline updates
 * @param agentId - The agent ID to subscribe to updates for
 * @returns The latest pipeline data and connection status
 */
export const useSentimentPipelineSocket = (agentId: string) => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [pipelineData, setPipelineData] = useState<SentimentPipelineSocketData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  
  // Store agent ID in ref to prevent unnecessary effect triggers
  const agentIdRef = useRef<string>(agentId);
  
  // Update ref if agent ID changes
  useEffect(() => {
    agentIdRef.current = agentId;
  }, [agentId]);
  
  // Handler for pipeline updates
  const handlePipelineUpdate = useCallback((data: any) => {
    if (data && data.agent_id === agentIdRef.current) {
      setPipelineData(data);
      setLoading(false);
    }
  }, []);
  
  // Handle connection status changes
  const handleConnectionStatus = useCallback((connected: boolean, errorMsg?: string) => {
    setIsConnected(connected);
    if (!connected && errorMsg) {
      setError(errorMsg);
    } else {
      setError(null);
    }
  }, []);
  
  // Connect to WebSocket and subscribe to sentiment pipeline updates
  useEffect(() => {
    if (!agentId) return;
    
    console.log(`Subscribing to sentiment pipeline updates for agent ${agentId}`);
    setLoading(true);
    
    // Add event handlers
    webSocketService.addTopicHandler(WebSocketTopic.SENTIMENT_PIPELINE, handlePipelineUpdate);
    webSocketService.addConnectionStatusHandler(handleConnectionStatus);
    
    // Connect if not already connected
    if (!webSocketService.isConnectionOpen()) {
      webSocketService.connect(WebSocketTopic.SENTIMENT_PIPELINE)
        .then(() => {
          // Send subscription message for this specific agent
          webSocketService.send({
            type: 'subscribe',
            topic: WebSocketTopic.SENTIMENT_PIPELINE,
            data: { agent_id: agentId }
          });
        })
        .catch(err => {
          console.error('Failed to connect to WebSocket', err);
          setError('Failed to connect to WebSocket server');
          setLoading(false);
        });
    } else {
      // Already connected, just send subscription
      webSocketService.send({
        type: 'subscribe',
        topic: WebSocketTopic.SENTIMENT_PIPELINE,
        data: { agent_id: agentId }
      });
    }
    
    // Cleanup function
    return () => {
      webSocketService.removeTopicHandler(WebSocketTopic.SENTIMENT_PIPELINE, handlePipelineUpdate);
      webSocketService.removeConnectionStatusHandler(handleConnectionStatus);
      
      // Unsubscribe from this agent's updates
      if (webSocketService.isConnectionOpen()) {
        webSocketService.send({
          type: 'unsubscribe',
          topic: WebSocketTopic.SENTIMENT_PIPELINE,
          data: { agent_id: agentId }
        });
      }
    };
  }, [agentId, handlePipelineUpdate, handleConnectionStatus]);
  
  // Create a function to manually refresh data
  const refreshData = useCallback(() => {
    if (webSocketService.isConnectionOpen()) {
      webSocketService.send({
        type: 'request',
        topic: WebSocketTopic.SENTIMENT_PIPELINE,
        data: { agent_id: agentIdRef.current }
      });
      setLoading(true);
    }
  }, []);
  
  return {
    isConnected,
    pipelineData,
    error,
    loading,
    refreshData
  };
};

export default useSentimentPipelineSocket;
