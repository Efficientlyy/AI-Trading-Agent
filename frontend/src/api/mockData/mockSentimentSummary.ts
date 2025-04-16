// mockSentimentSummary.ts
import { SentimentSignal } from '../../types';

export const getMockSentimentSummary = async (): Promise<{ sentimentData: Record<string, SentimentSignal> }> => {
  return {
    sentimentData: {
      AAPL: { signal: 'buy', strength: 0.8 },
      MSFT: { signal: 'hold', strength: 0.5 },
      TSLA: { signal: 'sell', strength: 0.7 },
      GOOG: { signal: 'buy', strength: 0.6 },
      AMZN: { signal: 'sell', strength: 0.4 },
    }
  };
};
