import { createAuthenticatedClient } from './client';
import { SentimentSignal } from '../types';

export const sentimentApi = {
  getSentimentSummary: async (): Promise<{ sentimentData: Record<string, SentimentSignal> }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ sentimentData: Record<string, SentimentSignal> }>('/sentiment/summary');
    return response.data;
  },
};
