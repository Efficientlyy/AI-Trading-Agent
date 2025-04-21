import React from 'react';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { SelectedAssetProvider } from './context/SelectedAssetContext';

// This is a minimal test to verify that our react-router-dom mock works
// and that the SelectedAssetProvider can be used in tests
test('SelectedAssetProvider works with MemoryRouter', () => {
  const { container } = render(
    <SelectedAssetProvider>
      <MemoryRouter>
        <div data-testid="test-content">Test Content</div>
      </MemoryRouter>
    </SelectedAssetProvider>
  );
  
  expect(container.textContent).toBe('Test Content');
});
