import React from 'react';
import { render } from '@testing-library/react';
import { MemoryRouter, BrowserRouter, Routes, Route } from 'react-router-dom';

describe('React Router DOM imports', () => {
  test('MemoryRouter can be imported and used', () => {
    expect(MemoryRouter).toBeDefined();
    
    const { container } = render(
      <MemoryRouter>
        <div>Test Content</div>
      </MemoryRouter>
    );
    
    expect(container.textContent).toBe('Test Content');
  });
  
  test('BrowserRouter can be imported', () => {
    expect(BrowserRouter).toBeDefined();
  });
  
  test('Routes and Route can be imported', () => {
    expect(Routes).toBeDefined();
    expect(Route).toBeDefined();
  });
});
