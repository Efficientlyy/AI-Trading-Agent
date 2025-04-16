import { createAuthenticatedClient } from './client';

export const performanceApi = {
  getPerformanceMetrics: async () => {
    const client = createAuthenticatedClient();
    const response = await client.get('/performance/metrics');
    return response.data;
  },
};
