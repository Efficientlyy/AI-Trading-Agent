import axios from 'axios';
import { BacktestResult, BacktestResultSummary } from '../types/backtest'; // Import the type we just defined

// Base URL for the backend API - adjust if needed, potentially move to config
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

/**
 * Fetches the detailed results of a specific backtest from the backend.
 * 
 * @param backtestId The ID of the backtest to retrieve.
 * @param token The JWT authentication token.
 * @returns A promise that resolves with the BacktestResult data.
 */
export const fetchBacktestResult = async (backtestId: string, token: string): Promise<BacktestResult> => {
  if (!backtestId) {
    return Promise.reject(new Error('Backtest ID is required.'));
  }
  if (!token) {
    // In a real app, you might redirect to login or refresh the token
    return Promise.reject(new Error('Authentication token is required.'));
  }

  try {
    const response = await axios.get<BacktestResult>(`${API_BASE_URL}/backtest/${backtestId}/result`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    // The backend currently returns mock data, but the structure matches BacktestResult
    return response.data;
  } catch (error) {
    console.error('Error fetching backtest result:', error);
    // Enhance error handling based on Axios error structure
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Failed to fetch backtest result: ${error.response.data.detail || error.message}`);
    } else {
      throw new Error(`Failed to fetch backtest result: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
};

/**
 * Fetches the history of backtests run by the user.
 * 
 * @param token The JWT authentication token.
 * @param limit Optional limit for the number of results.
 * @param offset Optional offset for pagination.
 * @param strategyId Optional filter by strategy ID.
 * @returns A promise that resolves with an array of BacktestResultSummary objects.
 */
export const fetchBacktestHistory = async (
  token: string,
  limit: number = 10,
  offset: number = 0,
  strategyId?: string
): Promise<{ backtests: BacktestResultSummary[]; count: number }> => {
  if (!token) {
    return Promise.reject(new Error('Authentication token is required.'));
  }

  try {
    const params = new URLSearchParams();
    params.append('limit', String(limit));
    params.append('offset', String(offset));
    if (strategyId) {
      params.append('strategy_id', strategyId);
    }

    const response = await axios.get<{ backtests: BacktestResultSummary[]; count: number }>(
      `${API_BASE_URL}/api/backtest/history`,
      {
        headers: { Authorization: `Bearer ${token}` },
        params: params
      }
    );
    // Backend returns mock data, but structure matches BacktestResultSummary[]
    return response.data;
  } catch (error) {
    console.error('Error fetching backtest history:', error);
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(`Failed to fetch backtest history: ${error.response.data.detail || error.message}`);
    } else {
      throw new Error(`Failed to fetch backtest history: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
};

// You might add other backtest-related API calls here later, e.g.:
// - startBacktest
// - fetchBacktestStatus
