# Modular Dashboard Pattern for AI Trading Agent

## Overview
This document describes the modular, context-aware dashboard architecture used in the AI Trading Agent frontend. The goal is to ensure each dashboard widget is self-contained, easy to maintain, and can dynamically switch between mock and real data sources using a global toggle.

---

## Key Principles

### 1. **Self-Contained Components**
- Each dashboard widget (e.g., PortfolioSummary, RecentTrades, PerformanceMetrics) fetches and manages its own data and loading/error states.
- No data, loading, or handler props are required from parent components (except for context-driven props like `symbol` or `currentPrice`).

### 2. **Global Data Source Toggle**
- The app provides a global context (e.g., `DataSourceContext`) for switching between mock and real data.
- All modular widgets use this context to determine which API (mock or real) to call.

### 3. **Standardized API Layer**
- Each data type (portfolio, trades, performance, etc.) has both a real API client and a mock API module, with matching return types.
- Example:
  - `portfolioApi.getPortfolio()` (real)
  - `getMockPortfolio()` (mock)

### 4. **Consistent UI/UX**
- Loading and error states are handled within each widget for a unified user experience.
- No prop-drilling or redundant wiring is needed.

---

## Example: Modular Widget Implementation

```tsx
// PortfolioSummary.tsx
const PortfolioSummary: React.FC = () => {
  const { dataSource } = useDataSource();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    const fetchPortfolio = async () => {
      try {
        const data = dataSource === 'mock'
          ? await getMockPortfolio()
          : await portfolioApi.getPortfolio();
        if (isMounted) setPortfolio(data.portfolio);
      } catch (e) {
        if (isMounted) setPortfolio(null);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchPortfolio();
    return () => { isMounted = false; };
  }, [dataSource]);

  // ...render logic
};
```

---

## Integration Pattern

- In your dashboard layout (e.g., `Dashboard.tsx`), simply use each modular widget directly:

```tsx
<PortfolioSummary />
<PerformanceMetrics />
<RecentTrades onSymbolSelect={handleSymbolChange} selectedSymbol={selectedSymbol} />
<AssetAllocationChart onAssetSelect={handleSymbolChange} selectedAsset={selectedSymbol} />
<SentimentSummary onSymbolSelect={handleSymbolChange} selectedSymbol={selectedSymbol} />
<OrderManagement symbol={selectedSymbol} currentPrice={currentPrice} />
```

- No need to pass data or loading props. Each widget will update automatically based on the global data source toggle.

---

## Benefits
- **Maintainability:** Widgets are easy to update and test in isolation.
- **Extensibility:** New widgets can be added using the same pattern.
- **Consistency:** Unified loading, error, and data-fetching logic.
- **Rapid Prototyping:** Toggle between mock and real data for development and demos.

---

## Recommendations for Future Development
- Apply this pattern to any new dashboard widgets.
- Keep mock and real API signatures in sync for smooth toggling.
- Document any widget-specific context or prop requirements.
- Encourage contributors to follow this modular/context-aware approach.

---

## See Also
- `src/context/DataSourceContext.tsx` (for the global data source toggle)
- `src/api/mockData/` (for mock API modules)
- `src/api/` (for real API clients)
- `src/pages/Dashboard.tsx` (for integration example)
