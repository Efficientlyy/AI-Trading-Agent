import React from 'react';
import { useBacktestResult } from '../../hooks/useBacktestResult';
import { BacktestResult } from '../../types/backtest';
import { Loader, Alert, Paper, Title, Text, SimpleGrid, Group } from '@mantine/core';
import { IconAlertCircle, IconChartLine, IconReportMoney, IconScale } from '@tabler/icons-react';

interface BacktestResultDisplayProps {
  backtestId: string | null | undefined;
  // TODO: Get token from auth context instead of passing as prop
  token: string | null | undefined; 
}

// Helper to format numbers (currency, percentage, ratio)
const formatNumber = (num: number | undefined | null, type: 'currency' | 'percent' | 'ratio' | 'integer', decimals = 2): string => {
  if (num === undefined || num === null || isNaN(num)) {
    return '-';
  }
  switch (type) {
    case 'currency':
      return `$${num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
    case 'percent':
      return `${(num * 100).toFixed(decimals)}%`;
    case 'ratio':
      return num.toFixed(decimals);
    case 'integer':
        return Math.round(num).toString();
    default:
      return num.toString();
  }
};

const BacktestResultDisplay: React.FC<BacktestResultDisplayProps> = ({ backtestId, token }) => {
  const { data: result, isLoading, isError, error } = useBacktestResult(backtestId, token);

  if (isLoading) {
    return (
      <Group justify="center" mt="xl">
        <Loader size="lg" />
        <Text>Loading backtest results...</Text>
      </Group>
    );
  }

  if (isError) {
    return (
      <Alert icon={<IconAlertCircle size="1rem" />} title="Error!" color="red" mt="xl">
        Failed to load backtest results: {error?.message || 'Unknown error'}
      </Alert>
    );
  }

  if (!result) {
    return (
      <Alert icon={<IconAlertCircle size="1rem" />} title="No Data" color="yellow" mt="xl">
        No backtest data available for the specified ID.
      </Alert>
    );
  }

  const { config, performance_metrics, trade_metrics } = result;

  return (
    <Paper shadow="xs" p="md" radius="md" withBorder>
      <Title order={2} mb="lg">Backtest Results: {config.strategy_id} ({backtestId?.substring(0, 8)})</Title>

      {/* Performance Metrics Section */}
      <Paper withBorder p="md" radius="sm" mb="lg">
        <Group mb="md">
          <IconReportMoney size="1.5rem" />
          <Title order={4}>Performance Metrics</Title>
        </Group>
        <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
          <Text><strong>Total Return:</strong> {formatNumber(performance_metrics.total_return, 'percent')}</Text>
          <Text><strong>Annualized Return:</strong> {formatNumber(performance_metrics.annualized_return, 'percent')}</Text>
          <Text><strong>Sharpe Ratio:</strong> {formatNumber(performance_metrics.sharpe_ratio, 'ratio')}</Text>
          <Text><strong>Sortino Ratio:</strong> {formatNumber(performance_metrics.sortino_ratio, 'ratio')}</Text>
          <Text><strong>Max Drawdown:</strong> {formatNumber(performance_metrics.max_drawdown, 'percent')}</Text>
          <Text><strong>Volatility:</strong> {formatNumber(performance_metrics.volatility, 'percent')}</Text>
          <Text><strong>Calmar Ratio:</strong> {formatNumber(performance_metrics.calmar_ratio, 'ratio')}</Text>
          {/* Add Beta/Alpha if available */}
        </SimpleGrid>
      </Paper>

      {/* Trade Metrics Section */}
      <Paper withBorder p="md" radius="sm" mb="lg">
        <Group mb="md">
          <IconScale size="1.5rem" />
          <Title order={4}>Trade Metrics</Title>
        </Group>
        <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
          <Text><strong>Total Trades:</strong> {formatNumber(trade_metrics.total_trades, 'integer')}</Text>
          <Text><strong>Win Rate:</strong> {formatNumber(trade_metrics.win_rate, 'percent')}</Text>
          <Text><strong>Profit Factor:</strong> {formatNumber(trade_metrics.profit_factor, 'ratio')}</Text>
          <Text><strong>Avg Profit:</strong> {formatNumber(trade_metrics.average_profit, 'currency')}</Text>
          <Text><strong>Avg Loss:</strong> {formatNumber(trade_metrics.average_loss, 'currency')}</Text>
          <Text><strong>Largest Profit:</strong> {formatNumber(trade_metrics.largest_profit, 'currency')}</Text>
          <Text><strong>Largest Loss:</strong> {formatNumber(trade_metrics.largest_loss, 'currency')}</Text>
        </SimpleGrid>
      </Paper>

      {/* Chart Section (Placeholder) */}
      <Paper withBorder p="md" radius="sm" mb="lg">
        <Group mb="md">
          <IconChartLine size="1.5rem" />
          <Title order={4}>Equity Curve & Drawdown</Title>
        </Group>
        <Text c="dimmed">Charts will be implemented here...</Text>
        {/* TODO: Add Chart component using result.portfolio_history */}
      </Paper>

      {/* Configuration Section (Optional) */}
      {/* 
      <Paper withBorder p="md" radius="sm">
        <Title order={4} mb="md">Backtest Configuration</Title>
        <SimpleGrid cols={2}>
          <Text>Start Date: {config.start_date}</Text>
          <Text>End Date: {config.end_date}</Text>
          <Text>Initial Capital: {formatNumber(config.initial_capital, 'currency')}</Text>
          <Text>Symbols: {config.symbols.join(', ')}</Text>
          <Text>Timeframe: {config.timeframe}</Text>
        </SimpleGrid>
      </Paper> 
      */}

    </Paper>
  );
};

export default BacktestResultDisplay;
