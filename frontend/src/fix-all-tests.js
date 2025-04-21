/**
 * Comprehensive Test Fixer for AI Trading Agent
 * 
 * This script applies fixes to all test files in the project, addressing common
 * TypeScript errors and test failures.
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const config = {
  testDirs: [
    'src/api',
    'src/components',
    'src/context',
    'src/pages',
    'src/tests',
    'src/utils'
  ],
  testFilePattern: /\.(test|spec)\.(ts|tsx)$/,
  fixTypes: {
    mockImplementations: true,
    testSyntax: true,
    imports: true,
    assertions: true,
    localStorage: true,
    dateHandling: true
  },
  dryRun: false // Set to true to preview changes without applying them
};

// Common fixes to apply to test files
const commonFixes = {
  // Fix mock implementations
  mockImplementations: {
    pattern: /\.mock(Resolved|Rejected)Value\(([^)]+)\)/g,
    replacement: (match, type, value) => {
      return type === 'Resolved' 
        ? `.mockImplementation(() => Promise.resolve(${value}))`
        : `.mockImplementation(() => Promise.reject(${value}))`;
    }
  },
  
  // Fix test function syntax
  testSyntax: {
    pattern: /(\s+)test\(/g,
    replacement: '$1it('
  },
  
  // Fix missing await in async tests
  asyncAssertions: {
    pattern: /expect\(([^)]+\([^)]*\))\)\.rejects\./g,
    replacement: 'expect(await $1).rejects.'
  },
  
  // Fix incorrect mock imports
  mockImports: {
    pattern: /jest\.mock\(['"]([^'"]+)['"]\)(\s*;)?/g,
    replacement: (match, modulePath) => {
      // Add implementation for common modules
      if (modulePath.includes('/api/utils/errorHandling')) {
        return `jest.mock('${modulePath}', () => ({
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
      
      if (modulePath.includes('/api/utils/monitoring')) {
        return `jest.mock('${modulePath}', () => ({
  recordApiCall: jest.fn(),
  canMakeApiCall: jest.fn().mockReturnValue(true),
  recordCircuitBreakerResult: jest.fn(),
  getCircuitBreakerState: jest.fn().mockReturnValue({ state: 'closed', remainingTimeMs: 0 }),
  resetCircuitBreaker: jest.fn()
}));`;
      }
      
      return match;
    }
  },
  
  // Fix localStorage mocks
  localStorage: {
    pattern: /localStorage\.(getItem|setItem)/g,
    check: (content) => content.includes('localStorage') && !content.includes('mockLocalStorage'),
    addToTop: `// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
  key: jest.fn(),
  length: 0
};

// Replace global localStorage with mock
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true
});
`
  },
  
  // Fix Date mocks
  dateHandling: {
    pattern: /new Date\(\)/g,
    check: (content) => content.includes('new Date()') && !content.includes('mockDate'),
    addToTop: `// Mock Date for consistent test results
const mockDate = new Date('2023-01-01T00:00:00Z');
jest.spyOn(global, 'Date').mockImplementation(() => mockDate);
`
  }
};

// Find all test files
const findTestFiles = () => {
  const testFiles = [];
  
  const searchDir = (dir) => {
    if (!fs.existsSync(dir)) return;
    
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const itemPath = path.join(dir, item);
      const stats = fs.statSync(itemPath);
      
      if (stats.isDirectory()) {
        searchDir(itemPath);
      } else if (stats.isFile() && config.testFilePattern.test(item)) {
        testFiles.push(itemPath);
      }
    }
  };
  
  for (const dir of config.testDirs) {
    searchDir(dir);
  }
  
  return testFiles;
};

// Apply fixes to a file
const applyFixes = (filePath) => {
  const relativePath = path.relative(process.cwd(), filePath);
  console.log(`Processing: ${relativePath}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  let modified = false;
  const appliedFixes = [];
  
  // Apply regex-based fixes
  for (const [fixName, fix] of Object.entries(commonFixes)) {
    // Skip if this fix type is disabled
    if (config.fixTypes[fixName] === false) continue;
    
    // For fixes that add content to the top of the file
    if (fix.check && fix.addToTop && fix.check(content)) {
      // Find the first import statement
      const importIndex = content.indexOf('import ');
      if (importIndex !== -1) {
        // Find the end of the import section
        let endOfImports = content.indexOf('\n\n', importIndex);
        if (endOfImports === -1) endOfImports = content.indexOf('\n', importIndex);
        if (endOfImports === -1) endOfImports = importIndex;
        
        // Insert the new content after the imports
        const newContent = content.slice(0, endOfImports + 1) + 
                          fix.addToTop + 
                          content.slice(endOfImports + 1);
        
        if (newContent !== content) {
          content = newContent;
          modified = true;
          appliedFixes.push(fixName);
        }
      }
    }
    
    // For regex-based replacements
    if (fix.pattern && fix.replacement) {
      const originalContent = content;
      content = content.replace(fix.pattern, fix.replacement);
      
      if (content !== originalContent) {
        modified = true;
        if (!appliedFixes.includes(fixName)) {
          appliedFixes.push(fixName);
        }
      }
    }
  }
  
  // Apply file-specific fixes based on filename
  const fileName = path.basename(filePath);
  
  // Save changes if modified
  if (modified && !config.dryRun) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`  Fixed: ${appliedFixes.join(', ')}`);
    return true;
  } else if (modified) {
    console.log(`  Would fix: ${appliedFixes.join(', ')} (dry run)`);
    return false;
  } else {
    console.log('  No changes needed');
    return false;
  }
};

// Run tests for a specific file
const runTest = (filePath) => {
  const relativePath = path.relative(process.cwd(), filePath);
  console.log(`Testing: ${relativePath}`);
  
  try {
    execSync(`npx jest ${relativePath} --no-cache`, { stdio: 'pipe' });
    console.log(`  ✅ Test passed`);
    return true;
  } catch (error) {
    console.log(`  ❌ Test failed`);
    return false;
  }
};

// Main function
const main = () => {
  console.log('=== AI Trading Agent Comprehensive Test Fixer ===');
  console.log(`Mode: ${config.dryRun ? 'Dry Run (preview only)' : 'Fix'}`);
  
  // Find all test files
  const testFiles = findTestFiles();
  console.log(`Found ${testFiles.length} test files`);
  
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
  
  if (!config.dryRun && fixedFiles > 0) {
    console.log(`\n=== Running Tests ===`);
    let passingTests = 0;
    
    // Only test files that were fixed
    for (const file of testFiles) {
      if (runTest(file)) {
        passingTests++;
      }
    }
    
    console.log(`\n=== Test Results ===`);
    console.log(`Passing tests: ${passingTests}/${testFiles.length}`);
    console.log(`Pass rate: ${(passingTests / testFiles.length * 100).toFixed(1)}%`);
  }
  
  console.log(`\n=== Next Steps ===`);
  console.log('1. Run all tests: npm test -- --watchAll=false');
  console.log('2. Check for remaining TypeScript errors: npx tsc --noEmit');
};

// Run the script
main();
