import { createAuthenticatedClient } from './client';
import { BacktestParams, BacktestResult, PerformanceMetrics } from '../types';

export const backtestApi = {
  startBacktest: async (params: BacktestParams): Promise<{ job_id: string, message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.post<{ job_id: string, message: string }>('/backtest/start', params);
    return response.data;
  },
  
  getBacktestStatus: async (jobId: string): Promise<{ status: string, progress: number, message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ status: string, progress: number, message: string }>(`/backtest/status/${jobId}`);
    return response.data;
  },
  
  getBacktestResults: async (jobId: string): Promise<BacktestResult> => {
    const client = createAuthenticatedClient();
    const response = await client.get<BacktestResult>(`/backtest/results/${jobId}`);
    return response.data;
  },
  
  getAllBacktests: async (): Promise<{ backtests: BacktestResult[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ backtests: BacktestResult[] }>('/backtest/list');
    return response.data;
  },
  
  getPerformanceMetrics: async (): Promise<PerformanceMetrics> => {
    const client = createAuthenticatedClient();
    const response = await client.get<PerformanceMetrics>('/metrics');
    return response.data;
  },
};
