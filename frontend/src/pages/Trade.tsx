import React, { useEffect, useState, useCallback } from "react";
import AssetSelector from "../components/trading/AssetSelector";
import { useSelectedAsset } from '../context/SelectedAssetContext';
import OrderBook from "../components/trading/OrderBook";
import OrderEntryForm from "../components/trading/OrderEntryForm";
import OrderManagement from "../components/trading/OrderManagement";
import PositionDetails from "../components/trading/PositionDetails";
import TradeHistory from "../components/trading/TradeHistory";
import TechnicalAnalysisChart from "../components/dashboard/TechnicalAnalysisChart";
import { portfolioApi } from '../api/portfolio';
import { marketApi } from '../api/market';
import { tradesApi } from '../api/trades';

import { Portfolio, Position, OHLCV } from '../types';

// --- Toast feedback ---
const Toast: React.FC<{ message: string; type: 'success' | 'error'; onClose: () => void }> = ({ message, type, onClose }) => (
  <div className={`fixed top-5 right-5 z-50 px-4 py-2 rounded shadow-lg text-white ${type === 'success' ? 'bg-green-600' : 'bg-red-600'}`}>
    {message}
    <button className="ml-3 text-lg font-bold" onClick={onClose}>×</button>
  </div>
);

const Trade: React.FC = () => {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const { symbol: selectedSymbol, setSymbol: setSelectedSymbol } = useSelectedAsset();
  const [trades, setTrades] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [position, setPosition] = useState<Position | null>(null);
  const [orderBook, setOrderBook] = useState<any>(null);
  const [orderBookLoading, setOrderBookLoading] = useState(false);
  const [orderBookError, setOrderBookError] = useState<string | null>(null);
  const [ohlcv, setOhlcv] = useState<OHLCV[] | null>(null);
  const [ohlcvLoading, setOhlcvLoading] = useState(false);
  const [ohlcvError, setOhlcvError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [chartTimeframe, setChartTimeframe] = useState<'1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w'>('1d');

  // Fetch portfolio and available assets
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const [{ portfolio }, { assets }] = await Promise.all([
          portfolioApi.getPortfolio(),
          marketApi.getAssets()
        ]);
        setPortfolio(portfolio);
        const symbols = assets.map((a: any) => a.symbol || a.ticker || a.name);
        setAvailableSymbols(symbols);
        setSelectedSymbol(symbols[0] || "");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Fetch trades, position, order book, and OHLCV when selected symbol or timeframe changes
  useEffect(() => {
    if (!selectedSymbol) return;
    // Fetch trades
    (async () => {
      try {
        const { trades } = await tradesApi.getRecentTrades();
        setTrades(trades.filter((t: any) => t.symbol === selectedSymbol));
      } catch {
        setTrades([]);
      }
    })();
    // Fetch position
    if (portfolio && portfolio.positions) {
      setPosition(portfolio.positions[selectedSymbol] || null);
    }
    // Fetch order book (mocked for now)
    setOrderBookLoading(true);
    setOrderBookError(null);
    setTimeout(() => {
      // Simulate order book data
      setOrderBook({
        bids: [
          { price: 99.5, size: 10 },
          { price: 99.4, size: 8 },
          { price: 99.3, size: 5 }
        ],
        asks: [
          { price: 100.1, size: 7 },
          { price: 100.2, size: 12 },
          { price: 100.3, size: 6 }
        ]
      });
      setOrderBookLoading(false);
    }, 500);
    // Fetch OHLCV
    setOhlcvLoading(true);
    setOhlcvError(null);
    (async () => {
      try {
        const now = new Date();
        const start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
        const end = now.toISOString();
        const { data } = await marketApi.getHistoricalData({ symbol: selectedSymbol, start, end, timeframe: chartTimeframe });
        setOhlcv(data);
      } catch (err) {
        setOhlcvError('Failed to fetch chart data');
        setOhlcv(null);
      } finally {
        setOhlcvLoading(false);
      }
    })();
  }, [selectedSymbol, portfolio, chartTimeframe]);

  // Handle order submission with feedback
  const handleSubmitOrder = useCallback(async (order: any) => {
    try {
      await portfolioApi.createOrder(order);
      setToast({ message: 'Order placed successfully!', type: 'success' });
      // Refresh portfolio and trades after order
      if (portfolio) {
        const { portfolio: updatedPortfolio } = await portfolioApi.getPortfolio();
        setPortfolio(updatedPortfolio);
      }
      const { trades: updatedTrades } = await tradesApi.getRecentTrades();
      setTrades(updatedTrades.filter((t: any) => t.symbol === selectedSymbol));
    } catch (err: any) {
      setToast({ message: err?.message || 'Order failed', type: 'error' });
    }
  }, [portfolio, selectedSymbol]);

  // Handle toast close
  const handleToastClose = () => setToast(null);

  // Handle chart timeframe change
  const handleTimeframeChange = (tf: typeof chartTimeframe) => setChartTimeframe(tf);

  if (loading) {
    return <div className="p-8 text-center text-gray-500">Loading trading interface...</div>;
  }

  return (
    <div className="trade-page-container p-6">
      {toast && <Toast message={toast.message} type={toast.type} onClose={handleToastClose} />}
      <h1 className="text-2xl font-bold mb-6">Trading Interface</h1>
      {/* Asset Selector */}
      <section className="mb-4">
        <AssetSelector assets={availableSymbols} />
      </section>
      {/* Live Price Chart */}
      <section className="mb-4">
        <TechnicalAnalysisChart
          symbol={selectedSymbol}
          data={ohlcv}
          isLoading={ohlcvLoading}
          error={ohlcvError}
          onTimeframeChange={handleTimeframeChange}
          timeframe={chartTimeframe}
        />
      </section>
      {/* Order Entry Form */}
      <section className="mb-4">
        <OrderEntryForm
          portfolio={portfolio}
          availableSymbols={availableSymbols}
          selectedSymbol={selectedSymbol}
          onSymbolChange={setSelectedSymbol}
          onSubmitOrder={handleSubmitOrder}
        />
      </section>
      {/* Order Book & Trade History */}
      <section className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <OrderBook symbol={selectedSymbol} orderBook={orderBook} isLoading={orderBookLoading} error={orderBookError} />
        <TradeHistory trades={trades} />
      </section>
      {/* Open Orders & Order History */}
      <section className="mb-4">
        <OrderManagement symbol={selectedSymbol} />
      </section>
      {/* Position Details */}
      <section className="mb-4">
        <PositionDetails symbol={selectedSymbol} position={position} />
      </section>
      {/* Trade Confirmation & Feedback (now handled by toast) */}
    </div>
  );
};

export default Trade;
