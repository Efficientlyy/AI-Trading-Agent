/**
 * Enhanced script to check for common TypeScript errors in the codebase
 * with special focus on test files and mocking issues
 * 
 * This script helps identify and fix common TypeScript errors in test files,
 * particularly related to mocking and Jest compatibility.
 */
const fs = require('fs');
const path = require('path');

// List of directories to check
const dirsToCheck = [
  'src/api',
  'src/components',
  'src/context',
  'src/pages',
  'src/tests',
  'src/types',
  'src/utils'
];

// Patterns for common test-related issues
const testPatterns = {
  // Pattern for incorrect mock implementations
  incorrectMockImplementation: /\.mock(Resolved|Rejected)Value\(/g,
  // Pattern for missing mock implementations
  missingMockImplementation: /jest\.mock\([^)]+\)\s*;/g,
  // Pattern for incorrect mock return values
  incorrectMockReturnValue: /\.mockReturnValue\([^)]*Promise\./g,
  // Pattern for incorrect test function usage
  incorrectTestFunction: /\s+test\(/g,
  // Pattern for incorrect expect usage
  incorrectExpectUsage: /expect\([^)]+\)\.toHaveBeenCalledWith\([^)]*\)\)/g,
};

// Function to recursively get all files in a directory
function getAllFiles(dirPath, arrayOfFiles = []) {
  const files = fs.readdirSync(dirPath);

  files.forEach(file => {
    const filePath = path.join(dirPath, file);
    if (fs.statSync(filePath).isDirectory()) {
      arrayOfFiles = getAllFiles(filePath, arrayOfFiles);
    } else {
      if (filePath.endsWith('.ts') || filePath.endsWith('.tsx')) {
        arrayOfFiles.push(filePath);
      }
    }
  });

  return arrayOfFiles;
}

// Function to check for common TypeScript errors
function checkForErrors(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const errors = [];
  const warnings = [];
  const fixes = [];

  // Check for duplicate exports
  const exportRegex = /export\s+(const|let|var|function|class|interface|type)\s+(\w+)/g;
  const exports = {};
  let match;
  while ((match = exportRegex.exec(content)) !== null) {
    const exportName = match[2];
    if (exports[exportName]) {
      errors.push(`Duplicate export: ${exportName}`);
    } else {
      exports[exportName] = true;
    }
  }

  // Check for missing imports
  const importRegex = /import\s+.*\s+from\s+['"](\S+)['"]\s*;?/g;
  while ((match = importRegex.exec(content)) !== null) {
    const importPath = match[1];
    if (importPath.startsWith('.')) {
      const resolvedPath = path.resolve(path.dirname(filePath), importPath);
      let fileExists = false;
      
      // Check for different extensions
      const possibleExtensions = ['.ts', '.tsx', '.js', '.jsx'];
      for (const ext of possibleExtensions) {
        if (fs.existsSync(`${resolvedPath}${ext}`)) {
          fileExists = true;
          break;
        }
      }
      
      if (!fileExists && !fs.existsSync(`${resolvedPath}/index.ts`) && !fs.existsSync(`${resolvedPath}/index.tsx`)) {
        errors.push(`Missing import: ${importPath}`);
      }
    }
  }

  // Special checks for test files
  if (filePath.includes('.test.') || filePath.includes('/__tests__/')) {
    // Check for problematic Jest mock implementations
    const mockResolvedValueRegex = /mock(Resolved|Rejected)Value\(/g;
    const mockImplementationRegex = /mockImplementation\(\(\)\s*=>\s*Promise\.(resolve|reject)\(/g;
    
    let mockResolvedCount = 0;
    while ((match = mockResolvedValueRegex.exec(content)) !== null) {
      mockResolvedCount++;
    }
    
    let mockImplementationCount = 0;
    while ((match = mockImplementationRegex.exec(content)) !== null) {
      mockImplementationCount++;
    }
    
    if (mockResolvedCount > 0 && mockImplementationCount === 0) {
      warnings.push(`Using mockResolvedValue/mockRejectedValue without mockImplementation may cause TypeScript errors`);
      fixes.push(`Consider replacing mockResolvedValue/mockRejectedValue with mockImplementation(() => Promise.resolve/reject())`);
    }
    
    // Check for missing NetworkError imports in error handling tests
    if (filePath.includes('errorHandling.test')) {
      if (content.includes('NetworkError') && !content.includes('import { NetworkError }') && !content.includes('NetworkError,')) {
        errors.push(`Using NetworkError without importing it`);
        fixes.push(`Add NetworkError to the imports from './errorHandling'`);
      }
    }
    
    // Check for incorrect mocking patterns
    if (content.includes('jest.mock') && content.includes('mockReturnValue') && !content.includes('mockImplementation')) {
      warnings.push(`Using mockReturnValue for functions that return promises can cause TypeScript errors`);
      fixes.push(`Consider using mockImplementation(() => Promise.resolve()) instead of mockReturnValue()`);
    }
  }

  return { errors, warnings, fixes };
}

/**
 * Function to fix common issues in test files
 * This function applies various fixes to test files to resolve TypeScript errors
 */
function fixTestFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  let fixed = false;
  const fixes = [];
  
  // Fix 1: Replace mockResolvedValue with mockImplementation
  const mockResolvedValueRegex = /\.mockResolvedValue\(([^)]+)\)/g;
  const fixedContent1 = content.replace(mockResolvedValueRegex, '.mockImplementation(() => Promise.resolve($1))');
  if (fixedContent1 !== content) {
    content = fixedContent1;
    fixed = true;
    fixes.push('Replaced mockResolvedValue with mockImplementation(() => Promise.resolve())'); 
  }
  
  // Fix 2: Replace mockRejectedValue with mockImplementation
  const mockRejectedValueRegex = /\.mockRejectedValue\(([^)]+)\)/g;
  const fixedContent2 = content.replace(mockRejectedValueRegex, '.mockImplementation(() => Promise.reject($1))');
  if (fixedContent2 !== content) {
    content = fixedContent2;
    fixed = true;
    fixes.push('Replaced mockRejectedValue with mockImplementation(() => Promise.reject())'); 
  }
  
  // Fix 3: Add NetworkError to imports if missing but used
  if (filePath.includes('errorHandling.test')) {
    if (content.includes('NetworkError') && !content.includes('import { NetworkError }') && !content.includes('NetworkError,')) {
      const importRegex = /(import \{ [^}]+ \} from '\.\/(errorHandling|[^']+)')/;
      const fixedContent3 = content.replace(importRegex, (match, p1) => {
        if (match.includes('errorHandling')) {
          return match.replace('import {', 'import { NetworkError,');
        } else {
          return `import { NetworkError } from './errorHandling';\n${match}`;
        }
      });
      if (fixedContent3 !== content) {
        content = fixedContent3;
        fixed = true;
        fixes.push('Added NetworkError to imports');
      }
    }
  }
  
  // Fix 4: Replace test() with it() for better Jest compatibility
  const testFunctionRegex = /(\s+)test\(/g;
  const fixedContent4 = content.replace(testFunctionRegex, '$1it(');
  if (fixedContent4 !== content) {
    content = fixedContent4;
    fixed = true;
    fixes.push('Replaced test() with it() for better Jest compatibility');
  }
  
  // Fix 5: Fix mock implementations for API modules
  if (filePath.includes('.test.') && content.includes('jest.mock(')) {
    // Check for simple mocks that need implementation details
    const simpleMockRegex = /jest\.mock\(['"](.*?)['"]\)(;|,)/g;
    const fixedContent5 = content.replace(simpleMockRegex, (match, modulePath) => {
      // Don't modify mocks that already have implementations
      if (match.includes('=>') || match.includes('() =>')) {
        return match;
      }
      
      // Add basic implementation for common modules
      if (modulePath.includes('/api/')) {
        return `jest.mock('${modulePath}', () => ({\n  // Auto-generated mock implementation\n  getAll: jest.fn().mockImplementation(() => Promise.resolve([])),\n  getById: jest.fn().mockImplementation(() => Promise.resolve({})),\n  create: jest.fn().mockImplementation(() => Promise.resolve({ id: 'mock-id' })),\n  update: jest.fn().mockImplementation(() => Promise.resolve(true)),\n  delete: jest.fn().mockImplementation(() => Promise.resolve(true))\n}))${match.endsWith(';') ? ';' : ','}`;
      }
      
      return match;
    });
    
    if (fixedContent5 !== content) {
      content = fixedContent5;
      fixed = true;
      fixes.push('Added implementation details to API module mocks');
    }
  }
  
  // Fix 6: Fix syntax errors in test files
  // Look for common syntax errors like missing commas, parentheses, etc.
  const syntaxErrorPatterns = [
    { regex: /\)\s+status:/g, replacement: '), status:' }, // Missing comma after closing parenthesis
    { regex: /\}\s+\}/g, replacement: '} }' }, // Missing comma between objects
    { regex: /\(\s*\)\s*\)/g, replacement: '()' }, // Double parentheses
  ];
  
  for (const pattern of syntaxErrorPatterns) {
    const fixedContentSyntax = content.replace(pattern.regex, pattern.replacement);
    if (fixedContentSyntax !== content) {
      content = fixedContentSyntax;
      fixed = true;
      fixes.push(`Fixed syntax error: ${pattern.regex}`);
    }
  }
  
  // Apply fixes if any were made
  if (fixed) {
    fs.writeFileSync(filePath, content, 'utf8');
    return fixes;
  }
  
  return null;
}

/**
 * Main function to check for and fix TypeScript errors
 */
function main() {
  let allErrors = [];
  let allWarnings = [];
  let allFixes = [];
  let autoFixedFiles = [];
  let testFilesFixed = 0;
  let totalTestFiles = 0;

  // Process command line arguments
  const args = process.argv.slice(2);
  const fixMode = args.includes('--fix');
  const testOnly = args.includes('--test-only');
  const specificFile = args.find(arg => arg.startsWith('--file='))?.split('=')[1];
  
  console.log('\n=== AI Trading Agent TypeScript Error Fixer ===');
  console.log(`Mode: ${fixMode ? 'Fix' : 'Check'} ${testOnly ? '(Test files only)' : '(All files)'}`);
  if (specificFile) {
    console.log(`Targeting specific file: ${specificFile}`);
  }

  dirsToCheck.forEach(dir => {
    const dirPath = path.join(process.cwd(), dir);
    if (fs.existsSync(dirPath)) {
      const files = getAllFiles(dirPath);
      files.forEach(file => {
        // Skip non-test files if test-only mode is enabled
        if (testOnly && !file.includes('.test.') && !file.includes('/__tests__/')) {
          return;
        }
        
        // Skip files that don't match the specific file pattern
        if (specificFile && !file.includes(specificFile)) {
          return;
        }
        
        // Count test files
        if (file.includes('.test.') || file.includes('/__tests__/')) {
          totalTestFiles++;
        }
        
        const { errors, warnings, fixes } = checkForErrors(file);
        
        if (errors.length > 0) {
          allErrors.push({ file, errors });
        }
        
        if (warnings.length > 0) {
          allWarnings.push({ file, warnings });
        }
        
        if (fixes.length > 0) {
          allFixes.push({ file, fixes });
        }
        
        // Auto-fix test files if fix mode is enabled
        if (fixMode && (file.includes('.test.') || file.includes('/__tests__/'))) {
          const appliedFixes = fixTestFile(file);
          if (appliedFixes) {
            autoFixedFiles.push({ file, fixes: appliedFixes });
            testFilesFixed++;
          }
        }
      });
    }
  });

  // Print summary
  console.log('\n=== Summary ===');
  console.log(`Total test files: ${totalTestFiles}`);
  console.log(`Files with errors: ${allErrors.length}`);
  console.log(`Files with warnings: ${allWarnings.length}`);
  console.log(`Files with suggested fixes: ${allFixes.length}`);
  if (fixMode) {
    console.log(`Test files auto-fixed: ${testFilesFixed}`);
  }

  // Print detailed reports
  if (allErrors.length > 0) {
    console.log('\n--- ERRORS ---');
    allErrors.forEach(({ file, errors }) => {
      console.log(`\nFile: ${file}`);
      errors.forEach(error => console.log(`  - ${error}`));
    });
  }
  
  if (allWarnings.length > 0) {
    console.log('\n--- WARNINGS ---');
    allWarnings.forEach(({ file, warnings }) => {
      console.log(`\nFile: ${file}`);
      warnings.forEach(warning => console.log(`  - ${warning}`));
    });
  }
  
  if (allFixes.length > 0) {
    console.log('\n--- SUGGESTED FIXES ---');
    allFixes.forEach(({ file, fixes }) => {
      console.log(`\nFile: ${file}`);
      fixes.forEach(fix => console.log(`  - ${fix}`));
    });
  }
  
  if (autoFixedFiles.length > 0) {
    console.log('\n--- AUTO-FIXED FILES ---');
    autoFixedFiles.forEach(({ file, fixes }) => {
      console.log(`\nFile: ${file}`);
      fixes.forEach(fix => console.log(`  - ${fix}`));
    });
  }
  
  if (allErrors.length === 0 && allWarnings.length === 0) {
    console.log('\nNo TypeScript errors or warnings found!');
  }
  
  // Provide next steps
  console.log('\n=== Next Steps ===');
  if (fixMode) {
    console.log('1. Run tests to verify fixes: npm test -- --watchAll=false');
    console.log('2. Check for remaining TypeScript errors: npx tsc --noEmit');
  } else {
    console.log('1. Run with --fix to automatically fix issues: node src/fix-typescript-errors.js --fix');
    console.log('2. Run with --test-only to focus on test files: node src/fix-typescript-errors.js --test-only');
    console.log('3. Target a specific file: node src/fix-typescript-errors.js --file=errorHandling.test.ts --fix');
  }
}

main();
