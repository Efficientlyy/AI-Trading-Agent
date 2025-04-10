import React, { useState, useEffect } from 'react';

export interface RiskCalculatorProps {
  symbol: string;
  currentPrice?: number;
}

interface RiskParameters {
  entryPrice: number;
  stopLossPrice: number;
  positionSize: number;
  riskPercentage: number;
  targetPrice: number;
}

const RiskCalculator: React.FC<RiskCalculatorProps> = ({ symbol, currentPrice = 0 }) => {
  const [params, setParams] = useState<RiskParameters>({
    entryPrice: currentPrice || 0,
    stopLossPrice: 0,
    positionSize: 1,
    riskPercentage: 2,
    targetPrice: 0
  });
  
  const [accountSize, setAccountSize] = useState<number>(10000);
  
  // Update entry price when currentPrice changes
  useEffect(() => {
    if (currentPrice > 0) {
      setParams(prev => ({
        ...prev,
        entryPrice: currentPrice,
        // Set default stop loss 5% below entry price
        stopLossPrice: prev.stopLossPrice === 0 ? currentPrice * 0.95 : prev.stopLossPrice,
        // Set default target 10% above entry price
        targetPrice: prev.targetPrice === 0 ? currentPrice * 1.1 : prev.targetPrice
      }));
    }
  }, [currentPrice]);
  
  // Calculate risk metrics
  const calculations = React.useMemo(() => {
    const { entryPrice, stopLossPrice, positionSize, targetPrice } = params;
    
    if (entryPrice <= 0 || stopLossPrice <= 0 || positionSize <= 0 || targetPrice <= 0) {
      return {
        riskAmount: 0,
        rewardAmount: 0,
        riskRewardRatio: 0,
        maxPositionSize: 0,
        dollarRisk: 0,
        dollarReward: 0
      };
    }
    
    // Calculate risk per share
    const riskPerShare = Math.abs(entryPrice - stopLossPrice);
    
    // Calculate reward per share
    const rewardPerShare = Math.abs(targetPrice - entryPrice);
    
    // Calculate risk-reward ratio
    const riskRewardRatio = rewardPerShare / riskPerShare;
    
    // Calculate dollar risk
    const dollarRisk = riskPerShare * positionSize;
    
    // Calculate dollar reward
    const dollarReward = rewardPerShare * positionSize;
    
    // Calculate max position size based on risk percentage
    const maxRiskAmount = accountSize * (params.riskPercentage / 100);
    const maxPositionSize = riskPerShare > 0 ? maxRiskAmount / riskPerShare : 0;
    
    return {
      riskAmount: riskPerShare,
      rewardAmount: rewardPerShare,
      riskRewardRatio,
      maxPositionSize: Math.floor(maxPositionSize),
      dollarRisk,
      dollarReward
    };
  }, [params, accountSize]);
  
  // Handle input changes
  const handleParamChange = (key: keyof RiskParameters, value: number) => {
    setParams(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Risk Calculator - {symbol}</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label htmlFor="account-size" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Account Size ($)
          </label>
          <input
            id="account-size"
            type="number"
            min="0"
            step="1000"
            value={accountSize}
            onChange={(e) => setAccountSize(parseFloat(e.target.value) || 0)}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
        <div>
          <label htmlFor="risk-percentage" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Risk Per Trade (%)
          </label>
          <input
            id="risk-percentage"
            type="number"
            min="0.1"
            max="100"
            step="0.1"
            value={params.riskPercentage}
            onChange={(e) => handleParamChange('riskPercentage', parseFloat(e.target.value) || 0)}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label htmlFor="entry-price" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Entry Price ($)
          </label>
          <input
            id="entry-price"
            type="number"
            min="0"
            step="0.01"
            value={params.entryPrice}
            onChange={(e) => handleParamChange('entryPrice', parseFloat(e.target.value) || 0)}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
        <div>
          <label htmlFor="stop-loss" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Stop Loss ($)
          </label>
          <input
            id="stop-loss"
            type="number"
            min="0"
            step="0.01"
            value={params.stopLossPrice}
            onChange={(e) => handleParamChange('stopLossPrice', parseFloat(e.target.value) || 0)}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label htmlFor="target-price" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Target Price ($)
          </label>
          <input
            id="target-price"
            type="number"
            min="0"
            step="0.01"
            value={params.targetPrice}
            onChange={(e) => handleParamChange('targetPrice', parseFloat(e.target.value) || 0)}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
        <div>
          <label htmlFor="position-size" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Position Size (units)
          </label>
          <input
            id="position-size"
            type="number"
            min="0"
            step="1"
            value={params.positionSize}
            onChange={(e) => handleParamChange('positionSize', parseFloat(e.target.value) || 0)}
            className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>
      </div>
      
      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md mb-4">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Risk Analysis</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Risk Per Share</p>
            <p className="text-sm font-medium">${calculations.riskAmount.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Reward Per Share</p>
            <p className="text-sm font-medium">${calculations.rewardAmount.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Risk/Reward Ratio</p>
            <p className="text-sm font-medium">1:{calculations.riskRewardRatio.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Max Position Size</p>
            <p className="text-sm font-medium">{calculations.maxPositionSize} units</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Dollar Risk</p>
            <p className={`text-sm font-medium ${calculations.dollarRisk > accountSize * (params.riskPercentage / 100) ? 'text-red-600 dark:text-red-400' : ''}`}>
              ${calculations.dollarRisk.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Dollar Reward</p>
            <p className="text-sm font-medium">${calculations.dollarReward.toFixed(2)}</p>
          </div>
        </div>
      </div>
      
      {/* Position Sizing Recommendation */}
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
        <h3 className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">Position Sizing Recommendation</h3>
        <p className="text-sm text-blue-600 dark:text-blue-400">
          Based on your risk settings ({params.riskPercentage}% of ${accountSize.toLocaleString()}), 
          you should trade no more than <strong>{calculations.maxPositionSize}</strong> units of {symbol}.
        </p>
        {calculations.dollarRisk > accountSize * (params.riskPercentage / 100) && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">
            Warning: Your current position size exceeds your risk tolerance.
          </p>
        )}
      </div>
    </div>
  );
};

export default RiskCalculator;
