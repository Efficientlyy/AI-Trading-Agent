export const performanceApi = {
  getPerformanceMetrics: async () => {
    // Always use mock data for performance metrics
    const { getMockPerformanceMetrics } = await import('./mockData/mockPerformanceMetrics');
    return getMockPerformanceMetrics();
  },
};
