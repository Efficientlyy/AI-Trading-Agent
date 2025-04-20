import React, { useState } from 'react';
import { 
  runStandardPerformanceTestSuite, 
  generatePerformanceReport,
  testMemoization,
  testBatchProcessing,
  comparePerformance
} from '../utils/performanceTesting';

const PerformanceTestPage: React.FC = () => {
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [customTestType, setCustomTestType] = useState<'memoization' | 'batching' | 'comparison'>('memoization');
  const [customTestConfig, setCustomTestConfig] = useState({
    iterations: 100,
    batchSize: 10,
    itemCount: 50,
    delay: 5
  });

  // Run standard performance test suite
  const handleRunStandardTests = async () => {
    setIsRunningTests(true);
    setReportMarkdown('Running standard performance tests...');
    
    try {
      const results = await runStandardPerformanceTestSuite();
      setTestResults(results);
      
      const report = generatePerformanceReport(results);
      setReportMarkdown(report);
    } catch (error) {
      console.error('Error running performance tests:', error);
      setReportMarkdown(`Error running performance tests: ${error}`);
    } finally {
      setIsRunningTests(false);
    }
  };

  // Run custom performance test
  const handleRunCustomTest = async () => {
    setIsRunningTests(true);
    setReportMarkdown('Running custom performance test...');
    
    try {
      const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
      
      let results: Record<string, any> = {};
      
      if (customTestType === 'memoization') {
        // Custom memoization test
        const testFn = async (input: string): Promise<string> => {
          await delay(customTestConfig.delay);
          return `Processed: ${input}`;
        };
        
        const memoResults = await testMemoization(
          testFn,
          ['test-input'],
          customTestConfig.iterations
        );
        
        results.customMemoizationTest = memoResults;
      } else if (customTestType === 'batching') {
        // Custom batch processing test
        const singleProcessor = async (item: number): Promise<number> => {
          await delay(customTestConfig.delay);
          return item * 2;
        };
        
        const batchProcessor = async (items: number[]): Promise<number[]> => {
          await delay(customTestConfig.delay * 2); // Slightly longer for batch
          return items.map(item => item * 2);
        };
        
        const testItems = Array.from({ length: customTestConfig.itemCount }, (_, i) => i);
        
        const batchResults = await testBatchProcessing(
          singleProcessor,
          batchProcessor,
          testItems,
          customTestConfig.batchSize
        );
        
        results.customBatchingTest = batchResults;
      } else if (customTestType === 'comparison') {
        // Custom comparison test
        const originalFn = async (input: string): Promise<string> => {
          await delay(customTestConfig.delay * 2);
          return `Original: ${input}`;
        };
        
        const optimizedFn = async (input: string): Promise<string> => {
          await delay(customTestConfig.delay);
          return `Optimized: ${input}`;
        };
        
        const comparisonResults = await comparePerformance(
          originalFn,
          optimizedFn,
          ['test-input'],
          customTestConfig.iterations
        );
        
        results.customComparisonTest = comparisonResults;
      }
      
      setTestResults(prevResults => ({
        ...prevResults,
        ...results
      }));
      
      const report = generatePerformanceReport(results);
      setReportMarkdown(report);
    } catch (error) {
      console.error('Error running custom performance test:', error);
      setReportMarkdown(`Error running custom performance test: ${error}`);
    } finally {
      setIsRunningTests(false);
    }
  };

  // Render markdown report as HTML
  const renderMarkdown = () => {
    if (!reportMarkdown) {
      return <p className="text-gray-500 dark:text-gray-400">No test results yet. Run a test to see results.</p>;
    }
    
    // Very simple markdown to HTML conversion for tables and headers
    const html = reportMarkdown
      .replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold mb-4">$1</h1>')
      .replace(/^## (.*$)/gm, '<h2 class="text-xl font-semibold mt-6 mb-3">$1</h2>')
      .replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold mt-4 mb-2">$1</h3>')
      .replace(/^#### (.*$)/gm, '<h4 class="text-md font-semibold mt-3 mb-2">$1</h4>')
      .replace(/^\| (.*) \|$/gm, '<tr>$1</tr>')
      .replace(/\| ---([^|]*)/g, '<th class="px-4 py-2 text-left border-b-2 border-gray-300 dark:border-gray-600">$1</th>')
      .replace(/\| ([^|]*)/g, '<td class="px-4 py-2 border-b border-gray-200 dark:border-gray-700">$1</td>')
      .replace(/<tr>(.*?)<\/tr>/g, '<tr>$1</tr>')
      .replace(/^(.+)$/gm, '<p>$1</p>')
      .replace(/<\/tr><tr>/g, '</tr>\n<tr>')
      .replace(/<\/th><th/g, '</th>\n<th')
      .replace(/<\/td><td/g, '</td>\n<td')
      .replace(/\n\n+/g, '\n\n')
      .replace(/\n<tr>/g, '<tr>')
      .replace(/<tr>(.*?)<\/tr>/g, (match) => {
        if (match.includes('<th')) {
          return `<thead>${match}</thead>`;
        }
        return match;
      })
      .replace(/<thead>(.*?)<\/thead><tr>/g, '<thead>$1</thead><tbody><tr>')
      .replace(/<\/tr>\n\n<h/g, '</tr></tbody>\n\n<h')
      .replace(/<tr>(.*?)<\/tr>/g, (match) => {
        if (!match.includes('<th') && !match.includes('<thead')) {
          return `<tr>${match.slice(4, -5)}</tr>`;
        }
        return match;
      })
      .replace(/<table>/g, '<table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">')
      .replace(/\|\n\n/g, '</table>\n\n');
    
    return <div className="prose dark:prose-invert max-w-none" dangerouslySetInnerHTML={{ __html: html }} />;
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">Performance Testing</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Test Controls */}
        <div className="col-span-1">
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">Test Controls</h2>
            
            {/* Standard Test Suite */}
            <div className="mb-6">
              <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Standard Test Suite</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                Run a comprehensive suite of performance tests to evaluate the system's optimization features.
              </p>
              <button
                onClick={handleRunStandardTests}
                disabled={isRunningTests}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded disabled:opacity-50"
              >
                {isRunningTests ? 'Running Tests...' : 'Run Standard Tests'}
              </button>
            </div>
            
            {/* Custom Test */}
            <div>
              <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Custom Test</h3>
              
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Test Type
                </label>
                <select
                  value={customTestType}
                  onChange={(e) => setCustomTestType(e.target.value as any)}
                  className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3"
                  disabled={isRunningTests}
                >
                  <option value="memoization">Memoization Test</option>
                  <option value="batching">Batch Processing Test</option>
                  <option value="comparison">Function Comparison</option>
                </select>
              </div>
              
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Iterations
                </label>
                <input
                  type="number"
                  value={customTestConfig.iterations}
                  onChange={(e) => setCustomTestConfig({...customTestConfig, iterations: parseInt(e.target.value)})}
                  className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3"
                  disabled={isRunningTests}
                  min={1}
                  max={1000}
                />
              </div>
              
              {customTestType === 'batching' && (
                <>
                  <div className="mb-3">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      value={customTestConfig.batchSize}
                      onChange={(e) => setCustomTestConfig({...customTestConfig, batchSize: parseInt(e.target.value)})}
                      className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3"
                      disabled={isRunningTests}
                      min={1}
                      max={100}
                    />
                  </div>
                  
                  <div className="mb-3">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Item Count
                    </label>
                    <input
                      type="number"
                      value={customTestConfig.itemCount}
                      onChange={(e) => setCustomTestConfig({...customTestConfig, itemCount: parseInt(e.target.value)})}
                      className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3"
                      disabled={isRunningTests}
                      min={1}
                      max={1000}
                    />
                  </div>
                </>
              )}
              
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Simulated Delay (ms)
                </label>
                <input
                  type="number"
                  value={customTestConfig.delay}
                  onChange={(e) => setCustomTestConfig({...customTestConfig, delay: parseInt(e.target.value)})}
                  className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3"
                  disabled={isRunningTests}
                  min={1}
                  max={1000}
                />
              </div>
              
              <button
                onClick={handleRunCustomTest}
                disabled={isRunningTests}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded disabled:opacity-50"
              >
                {isRunningTests ? 'Running Test...' : 'Run Custom Test'}
              </button>
            </div>
          </div>
        </div>
        
        {/* Test Results */}
        <div className="col-span-1 md:col-span-2">
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">Test Results</h2>
            
            <div className="overflow-x-auto">
              {renderMarkdown()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceTestPage;
