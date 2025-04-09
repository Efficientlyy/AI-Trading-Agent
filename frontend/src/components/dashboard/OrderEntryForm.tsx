import React, { useState, useEffect } from 'react';
import { Portfolio } from '../../types';

interface OrderEntryFormProps {
  portfolio: Portfolio | null;
  availableSymbols: string[];
  onSubmitOrder: (order: {
    symbol: string;
    side: 'buy' | 'sell';
    type: 'market' | 'limit' | 'stop' | 'stop_limit';
    quantity: number;
    price?: number;
    stop_price?: number;
    time_in_force: 'day' | 'gtc' | 'ioc' | 'fok';
  }) => void;
}

const OrderEntryForm: React.FC<OrderEntryFormProps> = ({
  portfolio,
  availableSymbols,
  onSubmitOrder
}) => {
  // Form state
  const [symbol, setSymbol] = useState<string>('');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop' | 'stop_limit'>('market');
  const [quantity, setQuantity] = useState<string>('');
  const [price, setPrice] = useState<string>('');
  const [stopPrice, setStopPrice] = useState<string>('');
  const [timeInForce, setTimeInForce] = useState<'day' | 'gtc' | 'ioc' | 'fok'>('day');
  
  // Validation state
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [estimatedCost, setEstimatedCost] = useState<number | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [currentPosition, setCurrentPosition] = useState<{quantity: number, marketValue: number} | null>(null);

  // Reset price fields when order type changes
  useEffect(() => {
    if (orderType === 'market') {
      setPrice('');
      setStopPrice('');
    } else if (orderType === 'limit') {
      setStopPrice('');
    } else if (orderType === 'stop') {
      setPrice('');
    }
  }, [orderType]);

  // Update current price and position when symbol changes
  useEffect(() => {
    if (symbol && portfolio?.positions?.[symbol]) {
      const position = portfolio.positions[symbol];
      setCurrentPrice(position.current_price);
      setCurrentPosition({
        quantity: position.quantity,
        marketValue: position.market_value
      });
    } else {
      setCurrentPrice(null);
      setCurrentPosition(null);
    }
  }, [symbol, portfolio]);

  // Calculate estimated cost
  useEffect(() => {
    if (quantity && !isNaN(parseFloat(quantity))) {
      const quantityValue = parseFloat(quantity);
      let priceValue = 0;
      
      if (orderType === 'market' && currentPrice) {
        priceValue = currentPrice;
      } else if ((orderType === 'limit' || orderType === 'stop_limit') && price && !isNaN(parseFloat(price))) {
        priceValue = parseFloat(price);
      } else if (orderType === 'stop' && stopPrice && !isNaN(parseFloat(stopPrice))) {
        priceValue = parseFloat(stopPrice);
      }
      
      if (priceValue > 0) {
        setEstimatedCost(quantityValue * priceValue);
      } else {
        setEstimatedCost(null);
      }
    } else {
      setEstimatedCost(null);
    }
  }, [quantity, price, stopPrice, orderType, currentPrice]);

  // Validate form
  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!symbol) {
      newErrors.symbol = 'Symbol is required';
    }
    
    if (!quantity || isNaN(parseFloat(quantity)) || parseFloat(quantity) <= 0) {
      newErrors.quantity = 'Valid quantity is required';
    }
    
    if (orderType === 'limit' || orderType === 'stop_limit') {
      if (!price || isNaN(parseFloat(price)) || parseFloat(price) <= 0) {
        newErrors.price = 'Valid limit price is required';
      }
    }
    
    if (orderType === 'stop' || orderType === 'stop_limit') {
      if (!stopPrice || isNaN(parseFloat(stopPrice)) || parseFloat(stopPrice) <= 0) {
        newErrors.stopPrice = 'Valid stop price is required';
      }
    }
    
    // Check if user has enough cash for buy orders
    if (side === 'buy' && estimatedCost && portfolio) {
      if (estimatedCost > portfolio.cash) {
        newErrors.general = 'Insufficient funds for this order';
      }
    }
    
    // Check if user has enough of the asset for sell orders
    if (side === 'sell' && quantity && symbol && portfolio?.positions?.[symbol]) {
      const quantityValue = parseFloat(quantity);
      const position = portfolio.positions[symbol];
      if (quantityValue > position.quantity) {
        newErrors.quantity = `Insufficient ${symbol} balance. You have ${position.quantity}`;
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      onSubmitOrder({
        symbol,
        side,
        type: orderType,
        quantity: parseFloat(quantity),
        price: price ? parseFloat(price) : undefined,
        stop_price: stopPrice ? parseFloat(stopPrice) : undefined,
        time_in_force: timeInForce
      });
      
      // Reset form
      setQuantity('');
      setPrice('');
      setStopPrice('');
    }
  };

  return (
    <div className="dashboard-widget col-span-1">
      <h2 className="text-lg font-semibold mb-3">Place Order</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="space-y-4">
          {/* General error message */}
          {errors.general && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded relative" role="alert">
              <span className="block sm:inline">{errors.general}</span>
            </div>
          )}
          
          {/* Symbol */}
          <div>
            <label htmlFor="symbol" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Symbol
            </label>
            <select
              id="symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className={`block w-full rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 shadow-sm focus:border-primary focus:ring-primary sm:text-sm ${errors.symbol ? 'border-red-500' : ''}`}
            >
              <option value="">Select a symbol</option>
              {availableSymbols.map((sym) => (
                <option key={sym} value={sym}>{sym}</option>
              ))}
            </select>
            {errors.symbol && <p className="mt-1 text-sm text-red-600">{errors.symbol}</p>}
          </div>
          
          {/* Current Position Info */}
          {currentPosition && (
            <div className="bg-blue-50 dark:bg-blue-900 p-2 rounded-md text-sm">
              <p className="font-medium text-blue-800 dark:text-blue-200">Current Position:</p>
              <div className="flex justify-between mt-1">
                <span className="text-blue-700 dark:text-blue-300">Quantity:</span>
                <span className="font-medium">{currentPosition.quantity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 8 })}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">Market Value:</span>
                <span className="font-medium">${currentPosition.marketValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
            </div>
          )}
          
          {/* Side */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Order Side
            </label>
            <div className="flex space-x-2">
              <button
                type="button"
                className={`flex-1 py-2 px-4 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  side === 'buy'
                    ? 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300 focus:ring-gray-500 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
                }`}
                onClick={() => setSide('buy')}
              >
                Buy
              </button>
              <button
                type="button"
                className={`flex-1 py-2 px-4 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  side === 'sell'
                    ? 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300 focus:ring-gray-500 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
                }`}
                onClick={() => setSide('sell')}
              >
                Sell
              </button>
            </div>
          </div>
          
          {/* Order Type */}
          <div>
            <label htmlFor="orderType" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Order Type
            </label>
            <select
              id="orderType"
              value={orderType}
              onChange={(e) => setOrderType(e.target.value as any)}
              className="block w-full rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
            >
              <option value="market">Market</option>
              <option value="limit">Limit</option>
              <option value="stop">Stop</option>
              <option value="stop_limit">Stop Limit</option>
            </select>
          </div>
          
          {/* Quantity */}
          <div>
            <label htmlFor="quantity" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Quantity
            </label>
            <div className="relative">
              <input
                type="text"
                id="quantity"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                className={`block w-full rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 shadow-sm focus:border-primary focus:ring-primary sm:text-sm ${errors.quantity ? 'border-red-500' : ''}`}
                placeholder="Enter quantity"
              />
              {currentPosition && side === 'sell' && (
                <button
                  type="button"
                  onClick={() => setQuantity(currentPosition.quantity.toString())}
                  className="absolute inset-y-0 right-0 px-3 flex items-center text-xs text-primary hover:text-primary-dark"
                >
                  MAX
                </button>
              )}
              {errors.quantity && <p className="mt-1 text-sm text-red-600">{errors.quantity}</p>}
            </div>
          </div>
          
          {/* Limit Price */}
          {(orderType === 'limit' || orderType === 'stop_limit') && (
            <div>
              <label htmlFor="price" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Limit Price
              </label>
              <div className="relative">
                <input
                  type="text"
                  id="price"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  className={`block w-full rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 shadow-sm focus:border-primary focus:ring-primary sm:text-sm ${errors.price ? 'border-red-500' : ''}`}
                  placeholder="Enter limit price"
                />
                {currentPrice && (
                  <button
                    type="button"
                    onClick={() => setPrice(currentPrice.toString())}
                    className="absolute inset-y-0 right-0 px-3 flex items-center text-xs text-primary hover:text-primary-dark"
                  >
                    CURRENT
                  </button>
                )}
                {errors.price && <p className="mt-1 text-sm text-red-600">{errors.price}</p>}
              </div>
            </div>
          )}
          
          {/* Stop Price */}
          {(orderType === 'stop' || orderType === 'stop_limit') && (
            <div>
              <label htmlFor="stopPrice" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Stop Price
              </label>
              <div className="relative">
                <input
                  type="text"
                  id="stopPrice"
                  value={stopPrice}
                  onChange={(e) => setStopPrice(e.target.value)}
                  className={`block w-full rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 shadow-sm focus:border-primary focus:ring-primary sm:text-sm ${errors.stopPrice ? 'border-red-500' : ''}`}
                  placeholder="Enter stop price"
                />
                {currentPrice && (
                  <button
                    type="button"
                    onClick={() => setStopPrice(currentPrice.toString())}
                    className="absolute inset-y-0 right-0 px-3 flex items-center text-xs text-primary hover:text-primary-dark"
                  >
                    CURRENT
                  </button>
                )}
                {errors.stopPrice && <p className="mt-1 text-sm text-red-600">{errors.stopPrice}</p>}
              </div>
            </div>
          )}
          
          {/* Time in Force */}
          <div>
            <label htmlFor="timeInForce" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Time in Force
            </label>
            <select
              id="timeInForce"
              value={timeInForce}
              onChange={(e) => setTimeInForce(e.target.value as any)}
              className="block w-full rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
            >
              <option value="day">Day</option>
              <option value="gtc">Good Till Cancelled</option>
              <option value="ioc">Immediate or Cancel</option>
              <option value="fok">Fill or Kill</option>
            </select>
          </div>
        </div>
        
        {/* Order Summary */}
        {estimatedCost !== null && (
          <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md mt-4">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Order Summary</h3>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Estimated {side === 'buy' ? 'Cost' : 'Proceeds'}:</span>
              <span className="font-medium">${estimatedCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            </div>
            {currentPrice && (
              <div className="flex justify-between text-sm mt-1">
                <span className="text-gray-600 dark:text-gray-400">Current Market Price:</span>
                <span className="font-medium">${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 })}</span>
              </div>
            )}
            {portfolio && (
              <div className="flex justify-between text-sm mt-1">
                <span className="text-gray-600 dark:text-gray-400">Available Cash:</span>
                <span className={`font-medium ${side === 'buy' && estimatedCost && estimatedCost > portfolio.cash ? 'text-red-500' : ''}`}>
                  ${portfolio.cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
            )}
          </div>
        )}
        
        {/* Submit Button */}
        <div className="mt-4">
          <button
            type="submit"
            className={`w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 ${
              side === 'buy'
                ? 'bg-green-600 hover:bg-green-700 focus:ring-green-500'
                : 'bg-red-600 hover:bg-red-700 focus:ring-red-500'
            }`}
          >
            {side === 'buy' ? 'Place Buy Order' : 'Place Sell Order'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default OrderEntryForm;
