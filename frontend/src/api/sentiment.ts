import { SentimentSignal } from '../types';

export const sentimentApi = {
  getSentimentSummary: async (): Promise<{ sentimentData: Record<string, SentimentSignal> }> => {
    // Always use mock data for sentiment summary
    const { getMockSentimentSummary } = await import('./mockData/mockSentimentSummary');
    return getMockSentimentSummary();
  },
};
