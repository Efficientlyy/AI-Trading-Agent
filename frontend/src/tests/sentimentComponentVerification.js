/**
 * Manual Verification Script for Sentiment Analysis Components
 * 
 * This script runs verification checks on our sentiment analysis components
 * to ensure our performance optimizations are working correctly.
 */

// Import paths for verification
console.log('=== SENTIMENT COMPONENT VERIFICATION ===');

// Verify component file structure
const fs = require('fs');
const path = require('path');

// Define paths to check
const componentPaths = [
  '../components/AgentFlowGrid/SentimentHistoricalChart.tsx',
  '../components/AgentFlowGrid/SymbolFilterControl.tsx',
  '../components/AgentFlowGrid/SignalQualityMetricsPanel.tsx',
  '../components/AgentFlowGrid/AdvancedAnalyticsTab.tsx',
  '../api/sentimentAnalyticsService.ts'
];

// Check if all required files exist
console.log('\nVerifying component file structure:');
let allFilesExist = true;
const absPathResults = {};

componentPaths.forEach(relativePath => {
  const absPath = path.resolve(__dirname, relativePath);
  const exists = fs.existsSync(absPath);
  absPathResults[relativePath] = { absPath, exists };
  
  console.log(`${exists ? '✅' : '❌'} ${relativePath}`);
  if (!exists) {
    allFilesExist = false;
  }
});

if (allFilesExist) {
  console.log('✅ All required component files exist.');
} else {
  console.error('❌ Some component files are missing. See details above.');
}

// Verify lazy loading in AdvancedAnalyticsTab
console.log('\nVerifying lazy loading implementation:');
const advancedTabPath = absPathResults['../components/AgentFlowGrid/AdvancedAnalyticsTab.tsx'].absPath;

if (fs.existsSync(advancedTabPath)) {
  const content = fs.readFileSync(advancedTabPath, 'utf8');
  
  const lazyLoadPattern = /const\s+\w+\s+=\s+lazy\(\s*\(\s*\)\s*=>\s*import\(\s*['"][^'"]+['"]\s*\)\s*\)/g;
  const lazyLoadMatches = content.match(lazyLoadPattern) || [];
  
  const suspensePattern = /<Suspense[^>]*>[\s\S]*?<\/Suspense>/g;
  const suspenseMatches = content.match(suspensePattern) || [];
  
  console.log(`Found ${lazyLoadMatches.length} lazy-loaded components`);
  console.log(`Found ${suspenseMatches.length} Suspense wrappers`);
  
  if (lazyLoadMatches.length >= 2 && suspenseMatches.length >= 1) {
    console.log('✅ Lazy loading is properly implemented.');
  } else {
    console.error('❌ Lazy loading implementation is incomplete or incorrect.');
  }
} else {
  console.error('❌ Cannot verify lazy loading: AdvancedAnalyticsTab.tsx not found.');
}

// Verify memoization in components
console.log('\nVerifying memoization implementation:');
const componentsToCheck = [
  {
    name: 'SentimentHistoricalChart',
    path: absPathResults['../components/AgentFlowGrid/SentimentHistoricalChart.tsx'].absPath,
    patterns: [
      { type: 'useMemo', regex: /useMemo\(\s*\(\s*\)\s*=>\s*{/g },
      { type: 'useCallback', regex: /useCallback\(\s*\(\s*\)\s*=>\s*{/g },
      { type: 'React.memo', regex: /export default React\.memo\(/g }
    ]
  },
  {
    name: 'SignalQualityMetricsPanel',
    path: absPathResults['../components/AgentFlowGrid/SignalQualityMetricsPanel.tsx'].absPath,
    patterns: [
      { type: 'useMemo', regex: /useMemo\(\s*\(\s*\)\s*=>\s*{/g },
      { type: 'useCallback', regex: /useCallback\(\s*\(\s*\)\s*=>\s*{/g },
      { type: 'React.memo', regex: /export default React\.memo\(/g }
    ]
  },
  {
    name: 'SymbolFilterControl',
    path: absPathResults['../components/AgentFlowGrid/SymbolFilterControl.tsx'].absPath,
    patterns: [
      { type: 'useMemo', regex: /useMemo\(\s*\(\s*\)\s*=>\s*{/g },
      { type: 'useCallback', regex: /useCallback\(\s*\(\s*\)\s*=>\s*{/g },
      { type: 'React.memo', regex: /export default React\.memo\(/g }
    ]
  }
];

componentsToCheck.forEach(component => {
  if (fs.existsSync(component.path)) {
    const content = fs.readFileSync(component.path, 'utf8');
    console.log(`\nChecking ${component.name}:`);
    
    let allPatternsFound = true;
    
    component.patterns.forEach(pattern => {
      const matches = content.match(pattern.regex) || [];
      const count = matches.length;
      
      console.log(`  ${count > 0 ? '✅' : '❌'} ${pattern.type}: ${count} occurrences`);
      
      if (count === 0) {
        allPatternsFound = false;
      }
    });
    
    if (allPatternsFound) {
      console.log(`✅ ${component.name} is properly memoized.`);
    } else {
      console.log(`❌ ${component.name} is missing some memoization optimizations.`);
    }
  } else {
    console.error(`❌ Cannot verify memoization: ${component.name} file not found.`);
  }
});

// Verify caching in sentimentAnalyticsService
console.log('\nVerifying caching implementation:');
const servicePath = absPathResults['../api/sentimentAnalyticsService.ts'].absPath;

if (fs.existsSync(servicePath)) {
  const content = fs.readFileSync(servicePath, 'utf8');
  
  const clientCachePattern = /clientCache\.get\(/g;
  const clientCacheMatches = content.match(clientCachePattern) || [];
  
  const clientCacheSetPattern = /clientCache\.set\(/g;
  const clientCacheSetMatches = content.match(clientCacheSetPattern) || [];
  
  console.log(`Found ${clientCacheMatches.length} cache retrieval operations`);
  console.log(`Found ${clientCacheSetMatches.length} cache storage operations`);
  
  if (clientCacheMatches.length >= 2 && clientCacheSetMatches.length >= 2) {
    console.log('✅ Client-side caching is properly implemented.');
  } else {
    console.error('❌ Client-side caching implementation is incomplete or incorrect.');
  }
} else {
  console.error('❌ Cannot verify caching: sentimentAnalyticsService.ts not found.');
}

// Verify Chart.js scale types and callback functions (the fixes we just made)
console.log('\nVerifying Chart.js configuration fixes:');
const metricsPath = absPathResults['../components/AgentFlowGrid/SignalQualityMetricsPanel.tsx'].absPath;

if (fs.existsSync(metricsPath)) {
  const content = fs.readFileSync(metricsPath, 'utf8');
  
  const linearScalePattern = /type:\s*['"]linear['"]\s*as\s*const/g;
  const linearScaleMatches = content.match(linearScalePattern) || [];
  
  const radialScalePattern = /type:\s*['"]radialLinear['"]\s*as\s*const/g;
  const radialScaleMatches = content.match(radialScalePattern) || [];
  
  const callbackPattern = /callback:\s*function\s*\(\s*tickValue:\s*string\s*\|\s*number\s*\)/g;
  const callbackMatches = content.match(callbackPattern) || [];
  
  console.log(`Found ${linearScaleMatches.length} linear scale type definitions`);
  console.log(`Found ${radialScaleMatches.length} radial scale type definitions`);
  console.log(`Found ${callbackMatches.length} properly typed tick callback functions`);
  
  if (linearScaleMatches.length >= 1 && radialScaleMatches.length >= 1 && callbackMatches.length >= 1) {
    console.log('✅ Chart.js configuration issues have been fixed.');
  } else {
    console.error('❌ Chart.js configuration fixes are incomplete or incorrect.');
  }
} else {
  console.error('❌ Cannot verify Chart.js fixes: SignalQualityMetricsPanel.tsx not found.');
}

// Overall verification summary
console.log('\n=== VERIFICATION SUMMARY ===');
if (
  allFilesExist && 
  (lazyLoadMatches?.length >= 2) && 
  (suspenseMatches?.length >= 1) && 
  (clientCacheMatches?.length >= 2) && 
  (clientCacheSetMatches?.length >= 2) &&
  (linearScaleMatches?.length >= 1) && 
  (radialScaleMatches?.length >= 1) && 
  (callbackMatches?.length >= 1)
) {
  console.log('🎉 All performance optimizations have been successfully implemented!');
} else {
  console.error('❌ Some performance optimizations are incomplete or incorrect. See details above.');
}
