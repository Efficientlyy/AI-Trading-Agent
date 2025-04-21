// Script to fix common test errors in the AI Trading Agent frontend
const fs = require('fs');
const path = require('path');

// Function to update monitoring.ts file to fix getApiCallMetrics export issues
function fixMonitoringFile() {
  const monitoringPath = path.join(process.cwd(), 'src/api/utils/monitoring.ts');
  
  if (!fs.existsSync(monitoringPath)) {
    console.error('Monitoring file not found at:', monitoringPath);
    return;
  }
  
  let content = fs.readFileSync(monitoringPath, 'utf8');
  
  // Check if there are duplicate getApiCallMetrics exports
  const getApiCallMetricsCount = (content.match(/export const getApiCallMetrics/g) || []).length;
  
  if (getApiCallMetricsCount > 1) {
    console.log('Found duplicate getApiCallMetrics exports in monitoring.ts. Fixing...');
    
    // Remove the duplicate export (keeping the first one)
    const firstOccurrence = content.indexOf('export const getApiCallMetrics');
    const secondOccurrence = content.indexOf('export const getApiCallMetrics', firstOccurrence + 1);
    
    if (secondOccurrence !== -1) {
      const startOfDuplicate = content.lastIndexOf('\n', secondOccurrence) + 1;
      const endOfFunction = content.indexOf('};', secondOccurrence) + 2;
      const duplicateContent = content.substring(startOfDuplicate, endOfFunction);
      
      // Replace the duplicate with a comment
      content = content.replace(duplicateContent, '// Duplicate getApiCallMetrics removed\n\n');
      
      fs.writeFileSync(monitoringPath, content);
      console.log('Fixed duplicate getApiCallMetrics in monitoring.ts');
    }
  } else {
    console.log('No duplicate getApiCallMetrics found in monitoring.ts');
  }
}

// Function to fix mock files issues
function fixMockFiles() {
  const mockDirs = [
    path.join(process.cwd(), 'src/api/trading/__mocks__'),
    path.join(process.cwd(), 'src/api/utils/__mocks__')
  ];
  
  mockDirs.forEach(dir => {
    if (fs.existsSync(dir)) {
      console.log(`Removing duplicate mock directory: ${dir}`);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
  
  console.log('Fixed duplicate mock directories');
}

// Function to update axios mock to include all required properties
function fixAxiosMock() {
  const axiosMockPath = path.join(process.cwd(), 'src/__mocks__/axios.ts');
  
  if (!fs.existsSync(axiosMockPath)) {
    console.error('Axios mock file not found at:', axiosMockPath);
    return;
  }
  
  const updatedAxiosMock = `// Mock implementation for axios
const axios = {
  create: jest.fn(() => ({
    get: jest.fn().mockResolvedValue({ data: {} }),
    post: jest.fn().mockResolvedValue({ data: {} }),
    put: jest.fn().mockResolvedValue({ data: {} }),
    delete: jest.fn().mockResolvedValue({ data: {} }),
    request: jest.fn().mockResolvedValue({ data: {} }),
    getUri: jest.fn(),
    defaults: {
      headers: {
        common: { Accept: 'application/json, text/plain, */*' },
        delete: {},
        get: {},
        head: {},
        post: { 'Content-Type': 'application/x-www-form-urlencoded' },
        put: { 'Content-Type': 'application/x-www-form-urlencoded' },
        patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
      }
    },
    interceptors: {
      request: { 
        use: jest.fn(), 
        eject: jest.fn(),
        clear: jest.fn()
      },
      response: { 
        use: jest.fn(), 
        eject: jest.fn(),
        clear: jest.fn()
      },
    },
    head: jest.fn().mockResolvedValue({ data: {} }),
    options: jest.fn().mockResolvedValue({ data: {} }),
    patch: jest.fn().mockResolvedValue({ data: {} }),
    postForm: jest.fn().mockResolvedValue({ data: {} }),
    putForm: jest.fn().mockResolvedValue({ data: {} }),
    patchForm: jest.fn().mockResolvedValue({ data: {} }),
  })),
  get: jest.fn().mockResolvedValue({ data: {} }),
  post: jest.fn().mockResolvedValue({ data: {} }),
  put: jest.fn().mockResolvedValue({ data: {} }),
  delete: jest.fn().mockResolvedValue({ data: {} }),
  request: jest.fn().mockResolvedValue({ data: {} }),
  isAxiosError: jest.fn().mockImplementation((error) => {
    return error && error.isAxiosError === true;
  }),
  getUri: jest.fn(),
  defaults: {
    headers: {
      common: { Accept: 'application/json, text/plain, */*' },
      delete: {},
      get: {},
      head: {},
      post: { 'Content-Type': 'application/x-www-form-urlencoded' },
      put: { 'Content-Type': 'application/x-www-form-urlencoded' },
      patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
    }
  },
  head: jest.fn().mockResolvedValue({ data: {} }),
  options: jest.fn().mockResolvedValue({ data: {} }),
  patch: jest.fn().mockResolvedValue({ data: {} }),
  postForm: jest.fn().mockResolvedValue({ data: {} }),
  putForm: jest.fn().mockResolvedValue({ data: {} }),
  patchForm: jest.fn().mockResolvedValue({ data: {} }),
};

// Add mockReturnValue and other Jest mock methods to the create function
axios.create.mockReturnValue = jest.fn();
axios.create.mockImplementation = jest.fn();
axios.create.mockResolvedValue = jest.fn();
axios.create.mockRejectedValue = jest.fn();

export default axios;
`;
  
  fs.writeFileSync(axiosMockPath, updatedAxiosMock);
  console.log('Fixed axios mock implementation');
}

// Function to fix React mock in Portfolio.test.tsx
function fixPortfolioTest() {
  const portfolioTestPath = path.join(process.cwd(), 'src/pages/Portfolio.test.tsx');
  
  if (!fs.existsSync(portfolioTestPath)) {
    console.error('Portfolio test file not found at:', portfolioTestPath);
    return;
  }
  
  let content = fs.readFileSync(portfolioTestPath, 'utf8');
  
  // Update the React mock to avoid out-of-scope variable references
  const reactMockRegex = /jest\.mock\('react'[\s\S]*?\}\);/;
  const updatedReactMock = `jest.mock('react', () => {
  const actualReact = jest.requireActual('react');
  return {
    ...actualReact,
    useState: jest.fn().mockImplementation(actualReact.useState),
    useEffect: jest.fn().mockImplementation(() => {}),
    useContext: jest.fn().mockImplementation(actualReact.useContext)
  };
});`;
  
  content = content.replace(reactMockRegex, updatedReactMock);
  
  fs.writeFileSync(portfolioTestPath, content);
  console.log('Fixed React mock in Portfolio.test.tsx');
}

// Main function to run all fixes
function main() {
  console.log('Starting to fix test errors in the AI Trading Agent frontend...');
  
  fixMonitoringFile();
  fixMockFiles();
  fixAxiosMock();
  fixPortfolioTest();
  
  console.log('All fixes applied. Try running the tests again.');
}

main();
