// Test if react-router-dom exists in node_modules without importing it
import { test, expect } from '@jest/globals';
const fs = require('fs');
const path = require('path');

test('react-router-dom package exists in node_modules', () => {
  const modulePath = path.resolve(__dirname, '../node_modules/react-router-dom');
  const exists = fs.existsSync(modulePath);
  expect(exists).toBe(true);
  
  if (exists) {
    const packageJsonPath = path.join(modulePath, 'package.json');
    const packageJsonExists = fs.existsSync(packageJsonPath);
    expect(packageJsonExists).toBe(true);
    
    if (packageJsonExists) {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      console.log('react-router-dom version:', packageJson.version);
    }
  }
});
