# AI Trading Agent: Smart Path for New Developers

This guide provides an optimized approach for new developers to understand and work with the AI Trading Agent project, with special attention to managing context window limitations when working with AI assistants.

## 1. Project Overview Map

### High-Level Architecture

The AI Trading Agent is organized into several key subsystems:

```
AI Trading Agent
├── Dashboard System - Web interface for monitoring and control
├── Sentiment Analysis System - Multi-source sentiment processing
├── Market Regime Detection - Market state classification
├── Risk Management - Position sizing and risk controls
├── Execution System - Order execution and exchange integration
├── Backtesting Framework - Strategy testing and optimization
└── Core Infrastructure - Logging, configuration, and utilities
```

### Critical Paths for Common Tasks

| Task | Key Directories | Entry Points | Related Components |
|------|----------------|--------------|-------------------|
| Dashboard Development | `/src/dashboard`, `/templates`, `/static` | `run_modern_dashboard.py` | Authentication, Data Service |
| Sentiment Analysis | `/src/sentiment`, `/config/sentiment_analysis.yaml` | `SentimentAnalysisManager` | News API, Social Media |
| Market Regime Work | `/src/ml/detection` | `RegimeDetectorFactory` | Ensemble Detector |
| Risk Management | `/src/risk` | `RiskManager` | Portfolio Manager |
| Strategy Development | `/src/strategy` | `BaseStrategy` | ML Models, Backtesting |

## 2. Core Component Guide

### 2.1 Dashboard System

The dashboard is a Flask-based web application with WebSocket support for real-time updates.

#### Data Sources

The dashboard supports two types of data sources:

1. **Real Data**: Connects to the actual trading system and displays real-time data
2. **Mock Data**: Uses generated mock data for development and testing

You can switch between data sources using the data source toggle in the dashboard settings (top-right corner). Mock data is automatically used as a fallback when real data is unavailable.

The data service layer (`src/dashboard/utils/data_service.py`) provides seamless switching between these sources, making it easy to develop and test without requiring a full trading system setup.

#### Key Files to Understand

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/dashboard/modern_dashboard.py` | Main dashboard class | Adding routes, API endpoints |
| `src/dashboard/utils/data_service.py` | Data access layer | Adding new data sources |
| `src/dashboard/utils/mock_data.py` | Mock data generation | Modifying test data |
| `src/dashboard/utils/auth.py` | Authentication system | User management, permissions |
| `templates/modern_dashboard.html` | Main template | Layout changes |
| `static/js/dashboard_optimizer.js` | Core JS functionality | UI behavior |
| `run_modern_dashboard.py` | Entry point | Startup configuration |

#### Files to Skip Initially

- Individual tab-specific JS files until you need to work on that specific tab
- Legacy dashboard files in `/archive/dashboard`
- Test files until you need to write tests

#### Dashboard Architecture

The dashboard follows a modular design with:

1. **ModernDashboard** class as the central controller
2. **DataService** for data access with caching
3. **Authentication** system with role-based access
4. **WebSocket** integration for real-time updates
5. **Tab-based** UI with specialized views

#### Common Dashboard Tasks

1. **Starting the Dashboard**:
   ```bash
   # Windows
   .\start_dashboard.ps1
   
   # Linux/macOS
   ./start_dashboard.sh
   ```

2. **Login Credentials**:
   - Admin: `admin`/`admin123`
   - Operator: `operator`/`operator123`
   - Viewer: `viewer`/`viewer123`

3. **Dashboard Tabs**:
   - Overview - System status and metrics
   - Market Regime - Current market conditions
   - Risk - Risk metrics and controls
   - Performance - Trading performance metrics
   - Logs - System logs and monitoring

### 2.2 Sentiment Analysis System

The Sentiment Analysis System processes data from multiple sources to generate actionable trading signals.

#### Key Files to Understand

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/sentiment/manager.py` | Central coordinator | Adding new sources |
| `src/sentiment/sources/` | Source implementations | Modifying specific sources |
| `src/sentiment/processors/` | Data processors | Changing processing logic |
| `config/sentiment_analysis.yaml` | Configuration | Adjusting parameters |

#### Sentiment Architecture

The system follows a pipeline architecture:

1. **Data Collection** from multiple sources (news, social media, etc.)
2. **Processing** with NLP techniques
3. **Aggregation** of signals with weighted scoring
4. **Integration** with trading strategies

#### Common Sentiment Tasks

1. **Adding a New Source**:
   - Create a new source class in `src/sentiment/sources/`
   - Register it in the `SentimentSourceFactory`
   - Add configuration in `config/sentiment_analysis.yaml`

2. **Tuning Sentiment Weights**:
   - Modify the weights in `config/sentiment_analysis.yaml`
   - Adjust the aggregation logic in `src/sentiment/aggregator.py`

### 2.3 Market Regime Detection

The Market Regime Detection system identifies market states to adapt trading strategies.

#### Key Files to Understand

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/ml/detection/base_regime_detector.py` | Base class | Creating new detectors |
| `src/ml/detection/ensemble_regime_detector.py` | Ensemble system | Modifying ensemble logic |
| `src/ml/detection/factory.py` | Detector factory | Registering new detectors |
| `config/regime_detection_config.yaml` | Configuration | Adjusting parameters |

#### Regime Detection Architecture

The system uses multiple detectors with an ensemble approach:

1. **Individual Detectors** (HMM, Volatility, Trend, Sentiment)
2. **Ensemble Detector** that combines individual predictions
3. **Confidence Scoring** to assess prediction reliability
4. **Regime-Specific Parameters** for trading strategies

#### Common Regime Detection Tasks

1. **Adding a New Detector**:
   - Create a new detector class inheriting from `BaseRegimeDetector`
   - Implement required methods
   - Register it in the `RegimeDetectorFactory`
   - Add configuration in `config/regime_detection_config.yaml`

2. **Tuning Ensemble Weights**:
   - Modify the weights in `config/regime_detection_config.yaml`
   - Adjust the ensemble logic in `src/ml/detection/ensemble_regime_detector.py`

### 2.4 Risk Management

The Risk Management system controls position sizing and risk exposure.

#### Key Files to Understand

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/risk/risk_manager.py` | Main risk controller | Overall risk logic |
| `src/risk/risk_budget_manager.py` | Budget allocation | Risk budget changes |
| `src/risk/drawdown_controller.py` | Drawdown protection | Drawdown limits |
| `config/risk.yaml` | Configuration | Risk parameters |

#### Risk Management Architecture

The system follows a hierarchical approach:

1. **System Risk Budget** at the top level
2. **Strategy Risk Budgets** allocated to strategies
3. **Market Risk Budgets** allocated to markets
4. **Asset Risk Budgets** allocated to assets
5. **Position Risk** for individual positions

#### Common Risk Management Tasks

1. **Adjusting Risk Limits**:
   - Modify the limits in `config/risk.yaml`
   - Update the risk budget allocation in `src/risk/risk_budget_manager.py`

2. **Adding New Risk Controls**:
   - Add new control logic in `src/risk/risk_manager.py`
   - Implement the control in the appropriate component

## 3. Exploration Efficiency Patterns

### Smart Workflows for AI Assistant Collaboration

When working with AI assistants on this project, follow these patterns to maximize efficiency:

#### Dashboard Development Workflow

1. **Start with structure understanding**:
   ```
   list_files src/dashboard
   list_code_definition_names src/dashboard/modern_dashboard.py
   ```

2. **Focus on specific components**:
   ```
   read_file src/dashboard/utils/data_service.py
   ```

3. **Make targeted changes**:
   ```
   replace_in_file src/dashboard/modern_dashboard.py
   ```

#### Sentiment Analysis Workflow

1. **Understand the configuration**:
   ```
   read_file config/sentiment_analysis.yaml
   ```

2. **Examine the manager**:
   ```
   list_code_definition_names src/sentiment/manager.py
   read_file src/sentiment/manager.py
   ```

3. **Look at specific sources as needed**:
   ```
   list_files src/sentiment/sources
   read_file src/sentiment/sources/news_api.py
   ```

#### Market Regime Workflow

1. **Start with the factory**:
   ```
   read_file src/ml/detection/factory.py
   ```

2. **Examine the base class**:
   ```
   read_file src/ml/detection/base_regime_detector.py
   ```

3. **Look at specific detectors as needed**:
   ```
   read_file src/ml/detection/hmm_regime_detector.py
   ```

### Context-Optimized Commands

To avoid wasting context window on unnecessary information:

1. **Use targeted directory listing**:
   - Instead of listing all files, list specific directories
   - Example: `list_files src/dashboard/utils` instead of `list_files src`

2. **Use code definition listing**:
   - Instead of reading entire files, list definitions first
   - Example: `list_code_definition_names src/dashboard/modern_dashboard.py`

3. **Use search for specific patterns**:
   - Instead of reading multiple files, search for patterns
   - Example: `search_files src/dashboard "class.*Dashboard"`

4. **Use targeted file reading**:
   - Read only the specific files you need
   - Skip test files and examples until needed

## 4. Task-Specific Guides

### Dashboard Feature Development

1. **Working with Data Sources**:
   - The dashboard has a data source toggle in the top-right corner
   - Options include "Mock" (default) and "Real"
   - During development, use "Mock" data to avoid dependencies on the full trading system
   - When testing with the actual trading system, switch to "Real" data
   - To modify mock data behavior, edit `src/dashboard/utils/mock_data.py`
   - To add new data sources, modify `src/dashboard/utils/data_service.py`

2. **Understand the tab structure**:
   - Each tab has a route in `modern_dashboard.py`
   - Each tab has a template in `templates/`
   - Each tab has JavaScript in `static/js/`

3. **Adding a new tab**:
   - Add a new route in `modern_dashboard.py`
   - Create a new template in `templates/`
   - Create new JavaScript in `static/js/`
   - Add the tab to the navigation in `templates/modern_dashboard.html`

4. **Adding a new visualization**:
   - Add the HTML structure in the template
   - Add the JavaScript to create the visualization
   - Add the data endpoint in `modern_dashboard.py`
   - Update the data service if needed
   - Ensure it works with both mock and real data sources

### Sentiment Analysis Update

1. **Adding a new sentiment source**:
   - Create a new source class in `src/sentiment/sources/`
   - Implement the required methods
   - Register it in the source factory
   - Add configuration in `config/sentiment_analysis.yaml`

2. **Updating sentiment processing**:
   - Modify the processor in `src/sentiment/processors/`
   - Update the aggregation logic if needed
   - Test the changes with `run_sentiment_tests.py`

### Adding a New Detector

1. **Create the detector class**:
   - Create a new file in `src/ml/detection/`
   - Inherit from `BaseRegimeDetector`
   - Implement the required methods

2. **Register the detector**:
   - Add it to the factory in `src/ml/detection/factory.py`
   - Add configuration in `config/regime_detection_config.yaml`

3. **Test the detector**:
   - Create a test file in `tests/ml/detection/`
   - Run the tests with pytest

## 5. Common Pitfalls and Optimization Tips

### Areas of High Complexity

1. **Ensemble Detector System**:
   - Complex interactions between detectors
   - Sophisticated confidence scoring
   - Adaptive weighting system

2. **Risk Budget Hierarchy**:
   - Multi-level risk allocation
   - Dynamic adjustments based on market conditions
   - Complex interaction with position sizing

3. **Sentiment Aggregation**:
   - Weighted combination of multiple sources
   - Time-series analysis of sentiment trends
   - Correlation with price movements

### Context Window Management

1. **When to start a new conversation**:
   - When switching to a different component
   - After completing a significant task
   - When the context window is getting full

2. **Information to include in new conversations**:
   - Brief summary of what you've learned
   - Specific files you need to work with
   - Clear task definition

3. **Efficient file exploration**:
   - Start with high-level structure
   - Drill down only into relevant components
   - Use search to find specific patterns

### Documentation References

Instead of including full documentation in the context, reference these key documents:

1. **Architecture**: `ARCHITECTURE.md`
2. **Dashboard**: `docs/DASHBOARD_ARCHITECTURE.md`, `docs/DASHBOARD_IMPLEMENTATION.md`
3. **Sentiment**: `docs/SENTIMENT_ANALYSIS_GUIDE.md`
4. **Market Regime**: `docs/market_regime_detection.md`
5. **Risk Management**: `docs/risk_management_implementation_guide.md`

## 6. Quick Reference: Key Components and Files

### Dashboard System

- **Entry Point**: `run_modern_dashboard.py`
- **Main Class**: `src/dashboard/modern_dashboard.py`
- **Templates**: `templates/`
- **Static Files**: `static/`
- **Data Service**: `src/dashboard/utils/data_service.py`
- **Authentication**: `src/dashboard/utils/auth.py`

### Sentiment Analysis

- **Manager**: `src/sentiment/manager.py`
- **Sources**: `src/sentiment/sources/`
- **Processors**: `src/sentiment/processors/`
- **Configuration**: `config/sentiment_analysis.yaml`

### Market Regime Detection

- **Factory**: `src/ml/detection/factory.py`
- **Base Class**: `src/ml/detection/base_regime_detector.py`
- **Ensemble**: `src/ml/detection/ensemble_regime_detector.py`
- **Configuration**: `config/regime_detection_config.yaml`

### Risk Management

- **Manager**: `src/risk/risk_manager.py`
- **Budget Manager**: `src/risk/risk_budget_manager.py`
- **Drawdown Controller**: `src/risk/drawdown_controller.py`
- **Configuration**: `config/risk.yaml`

## 7. Conclusion

This guide provides an optimized path for new developers to understand and work with the AI Trading Agent project. By following these patterns and focusing on the key components, you can efficiently navigate the codebase and make effective contributions without getting lost in unnecessary details.

Remember that the project is modular by design, allowing you to focus on specific components without needing to understand the entire system at once. Use this guide as a map to navigate the codebase efficiently and make targeted changes with confidence.
