/**
 * Test Diagnostics Tool for AI Trading Agent
 * 
 * This script helps identify which test files are passing and which are failing,
 * providing detailed diagnostic information to help fix TypeScript errors.
 */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

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
  testFilePattern: /\.test\.(ts|tsx)$/,
  outputFile: 'test-diagnostics-report.json',
  maxConcurrentTests: 1, // Run tests sequentially to avoid interference
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

// Run a single test and capture results
const runTest = (testFile) => {
  const relativePath = path.relative(process.cwd(), testFile);
  console.log(`Running test: ${relativePath}`);
  
  try {
    // Run the test with Jest directly to get more detailed output
    execSync(`npx jest ${relativePath} --no-cache`, { stdio: 'pipe' });
    
    return {
      file: relativePath,
      status: 'pass',
      error: null
    };
  } catch (error) {
    return {
      file: relativePath,
      status: 'fail',
      error: error.message
    };
  }
};

// Analyze test results
const analyzeResults = (results) => {
  const passingTests = results.filter(r => r.status === 'pass');
  const failingTests = results.filter(r => r.status === 'fail');
  
  const errorPatterns = {};
  
  // Analyze error patterns
  for (const test of failingTests) {
    const errorLines = test.error.split('\n');
    let errorType = 'unknown';
    
    // Try to categorize the error
    if (test.error.includes('TypeError:')) {
      errorType = 'TypeError';
    } else if (test.error.includes('SyntaxError:')) {
      errorType = 'SyntaxError';
    } else if (test.error.includes('Cannot find module')) {
      errorType = 'ModuleNotFound';
    } else if (test.error.includes('Property') && test.error.includes('does not exist')) {
      errorType = 'PropertyNotFound';
    } else if (test.error.includes('expect(')) {
      errorType = 'AssertionError';
    }
    
    if (!errorPatterns[errorType]) {
      errorPatterns[errorType] = [];
    }
    
    errorPatterns[errorType].push({
      file: test.file,
      error: errorLines.slice(0, 5).join('\n') // Just the first few lines
    });
  }
  
  return {
    summary: {
      total: results.length,
      passing: passingTests.length,
      failing: failingTests.length,
      passRate: `${(passingTests.length / results.length * 100).toFixed(1)}%`
    },
    passingTests: passingTests.map(t => t.file),
    failingTests: failingTests.map(t => t.file),
    errorPatterns
  };
};

// Generate recommendations based on analysis
const generateRecommendations = (analysis) => {
  const recommendations = [];
  
  // General recommendations
  if (analysis.summary.failing > 0) {
    recommendations.push('Run tests with verbose output to see detailed error messages: npm test -- --verbose');
  }
  
  // Error-specific recommendations
  if (analysis.errorPatterns.ModuleNotFound) {
    recommendations.push('Check import paths and make sure all dependencies are installed');
  }
  
  if (analysis.errorPatterns.TypeError) {
    recommendations.push('Review type definitions and ensure mock implementations match expected types');
  }
  
  if (analysis.errorPatterns.AssertionError) {
    recommendations.push('Check test assertions and mock return values');
  }
  
  if (analysis.errorPatterns.SyntaxError) {
    recommendations.push('Fix syntax errors in test files');
  }
  
  // Add specific file recommendations
  const mostFailingDir = getMostFailingDirectory(analysis.failingTests);
  if (mostFailingDir) {
    recommendations.push(`Focus on fixing tests in the ${mostFailingDir} directory first`);
  }
  
  return recommendations;
};

// Get the directory with the most failing tests
const getMostFailingDirectory = (failingTests) => {
  const dirCounts = {};
  
  for (const file of failingTests) {
    const dir = path.dirname(file);
    dirCounts[dir] = (dirCounts[dir] || 0) + 1;
  }
  
  let maxCount = 0;
  let maxDir = null;
  
  for (const dir in dirCounts) {
    if (dirCounts[dir] > maxCount) {
      maxCount = dirCounts[dir];
      maxDir = dir;
    }
  }
  
  return maxDir;
};

// Main function
const main = async () => {
  console.log('=== AI Trading Agent Test Diagnostics ===');
  
  // Find all test files
  const testFiles = findTestFiles();
  console.log(`Found ${testFiles.length} test files`);
  
  // Run tests
  const results = [];
  for (const testFile of testFiles) {
    results.push(runTest(testFile));
  }
  
  // Analyze results
  const analysis = analyzeResults(results);
  
  // Generate recommendations
  analysis.recommendations = generateRecommendations(analysis);
  
  // Save report
  fs.writeFileSync(config.outputFile, JSON.stringify(analysis, null, 2));
  
  // Print summary
  console.log('\n=== Test Results Summary ===');
  console.log(`Total tests: ${analysis.summary.total}`);
  console.log(`Passing: ${analysis.summary.passing}`);
  console.log(`Failing: ${analysis.summary.failing}`);
  console.log(`Pass rate: ${analysis.summary.passRate}`);
  
  console.log('\n=== Recommendations ===');
  for (const recommendation of analysis.recommendations) {
    console.log(`- ${recommendation}`);
  }
  
  console.log(`\nDetailed report saved to ${config.outputFile}`);
};

// Run the script
main().catch(error => {
  console.error('Error running diagnostics:', error);
  process.exit(1);
});
