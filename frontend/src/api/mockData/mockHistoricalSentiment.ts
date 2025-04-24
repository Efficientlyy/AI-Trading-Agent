// Helper function to generate mock historical sentiment data
export const getMockHistoricalSentiment = async (symbol: string, timeframe: string): Promise<{ timestamp: string; score: number }[]> => {
    console.log(`Fetching mock historical sentiment for ${symbol} (${timeframe})`);

    // Generate mock data points
    const data: { timestamp: string; score: number }[] = [];
    const now = new Date();
    let interval = 24 * 60 * 60 * 1000; // Default to 1 day interval

    switch (timeframe) {
        case '1D':
            interval = 60 * 60 * 1000; // 1 hour
            break;
        case '1W':
            interval = 6 * 60 * 60 * 1000; // 6 hours
            break;
        case '1M':
            interval = 24 * 60 * 60 * 1000; // 1 day
            break;
        case '1Y':
            interval = 7 * 24 * 60 * 60 * 1000; // 1 week
            break;
        default:
            interval = 24 * 60 * 60 * 1000; // Default to 1 day
    }

    const numPoints = timeframe === '1D' ? 24 : timeframe === '1W' ? 28 : timeframe === '1M' ? 30 : 52; // Approx points

    for (let i = numPoints; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * interval);
        // Generate a random sentiment score between -1 and 1
        const score = parseFloat((Math.random() * 2 - 1).toFixed(2));
        data.push({ timestamp: timestamp.toISOString(), score });
    }

    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));

    return data;
};
