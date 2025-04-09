import { createAuthenticatedClient } from './client';
import { Strategy } from '../types';

export const strategiesApi = {
  getStrategies: async (): Promise<{ strategies: Strategy[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ strategies: Strategy[] }>('/strategies');
    return response.data;
  },
  
  getStrategy: async (id: string): Promise<Strategy> => {
    const client = createAuthenticatedClient();
    const response = await client.get<Strategy>(`/strategies/${id}`);
    return response.data;
  },
  
  createStrategy: async (strategy: Omit<Strategy, 'id' | 'created_at' | 'updated_at'>): Promise<Strategy> => {
    const client = createAuthenticatedClient();
    const response = await client.post<Strategy>('/strategies', strategy);
    return response.data;
  },
  
  updateStrategy: async (id: string, strategy: Partial<Strategy>): Promise<Strategy> => {
    const client = createAuthenticatedClient();
    const response = await client.put<Strategy>(`/strategies/${id}`, strategy);
    return response.data;
  },
  
  deleteStrategy: async (id: string): Promise<{ status: string, strategy_id: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.delete<{ status: string, strategy_id: string }>(`/strategies/${id}`);
    return response.data;
  },
};
