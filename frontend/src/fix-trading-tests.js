/**
 * Trading API Test Fixer for AI Trading Agent
 * 
 * This script specifically targets and fixes common issues in the trading API test files.
 */
const fs = require('fs');
const path = require('path');

// Configuration
const config = {
  targetDir: path.join(process.cwd(), 'src', 'api', 'trading'),
  commonFixes: {
    // Fix mock implementations
    mockImplementation: {
      pattern: /\.mock(Resolved|Rejected)Value\(([^)]+)\)/g,
      replacement: (match, type, value) => {
        return type === 'Resolved' 
          ? `.mockImplementation(() => Promise.resolve(${value}))`
          : `.mockImplementation(() => Promise.reject(${value}))`;
      }
    },
    // Fix jest.mock calls
    jestMock: {
      pattern: /jest\.mock\(['"]([^'"]+)['"]\)(\s*;)?/g,
      replacement: (match, modulePath) => {
        // Add implementation for API modules
        if (modulePath.includes('/api/utils')) {
          return `jest.mock('${modulePath}', () => ({
  // Auto-generated mock implementation
  executeApiCall: jest.fn().mockImplementation((fn) => fn()),
  withRetry: jest.fn().mockImplementation((fn) => fn()),
  ApiError: jest.fn().mockImplementation((message, status, data, isRetryable) => ({
    message,
    status,
    data,
    isRetryable,
    name: 'ApiError'
  })),
  NetworkError: jest.fn().mockImplementation((message, isRetryable) => ({
    message,
    isRetryable,
    name: 'NetworkError'
  }))
}));`;
        }
        return match;
      }
    },
    // Fix test function calls
    testFunction: {
      pattern: /(\s+)test\(/g,
      replacement: '$1it('
    },
    // Fix missing await
    missingAwait: {
      pattern: /expect\(([^)]+\([^)]*\))\)\.rejects\./g,
      replacement: 'expect(await $1).rejects.'
    },
    // Fix incorrect assertions
    incorrectAssertions: {
      pattern: /expect\(([^)]+)\)\.toEqual\(([^)]+)\)/g,
      replacement: (match, actual, expected) => {
        // If comparing with empty arrays or objects, use toEqual
        if (expected.trim() === '[]' || expected.trim() === '{}') {
          return match;
        }
        // Otherwise use toBe for primitive values
        return `expect(${actual}).toBe(${expected})`;
      }
    }
  }
};

// Find all test files in the target directory
const findTestFiles = (dir) => {
  const files = [];
  
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stats = fs.statSync(itemPath);
    
    if (stats.isDirectory()) {
      files.push(...findTestFiles(itemPath));
    } else if (stats.isFile() && item.endsWith('.test.ts')) {
      files.push(itemPath);
    }
  }
  
  return files;
};

// Apply fixes to a file
const applyFixes = (filePath) => {
  console.log(`Processing: ${path.relative(process.cwd(), filePath)}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  let modified = false;
  const appliedFixes = [];
  
  // Apply common fixes
  for (const [fixName, fix] of Object.entries(config.commonFixes)) {
    const originalContent = content;
    content = content.replace(fix.pattern, fix.replacement);
    
    if (content !== originalContent) {
      modified = true;
      appliedFixes.push(fixName);
    }
  }
  
  // Apply file-specific fixes based on filename
  const fileName = path.basename(filePath);
  
  if (fileName === 'binanceTradingApi.test.ts') {
    // Fix Binance API specific issues
    if (content.includes('getPortfolio') && !content.includes('mockAccountInfo')) {
      content = content.replace(
        /describe\('Binance Trading API'/,
        `// Mock data for Binance API tests
const mockAccountInfo = {
  balances: [
    { asset: 'BTC', free: '1.0', locked: '0.0' },
    { asset: 'ETH', free: '10.0', locked: '0.0' },
    { asset: 'USDT', free: '1000.0', locked: '0.0' }
  ]
};

describe('Binance Trading API'`
      );
      modified = true;
      appliedFixes.push('addedMockData');
    }
  } else if (fileName === 'paperTradingApi.test.ts') {
    // Fix Paper Trading API specific issues
    if (content.includes('localStorage.getItem') && !content.includes('mockLocalStorage')) {
      content = content.replace(
        /beforeEach\(\(\) => {/,
        `// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn()
};

// Replace global localStorage with mock
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true
});

beforeEach(() => {`
      );
      modified = true;
      appliedFixes.push('addedMockLocalStorage');
    }
  }
  
  // Save changes if modified
  if (modified) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`  Fixed: ${appliedFixes.join(', ')}`);
    return true;
  } else {
    console.log('  No changes needed');
    return false;
  }
};

// Main function
const main = () => {
  console.log('=== AI Trading Agent Trading API Test Fixer ===');
  
  // Find all test files
  const testFiles = findTestFiles(config.targetDir);
  console.log(`Found ${testFiles.length} test files in ${config.targetDir}`);
  
  // Apply fixes
  let fixedFiles = 0;
  for (const file of testFiles) {
    if (applyFixes(file)) {
      fixedFiles++;
    }
  }
  
  console.log(`\n=== Summary ===`);
  console.log(`Total files: ${testFiles.length}`);
  console.log(`Fixed files: ${fixedFiles}`);
  
  console.log(`\n=== Next Steps ===`);
  console.log('1. Run tests to verify fixes: npm test -- --watchAll=false src/api/trading');
  console.log('2. Check for remaining TypeScript errors: npx tsc --noEmit');
};

// Run the script
main();
