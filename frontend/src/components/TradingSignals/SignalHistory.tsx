import React, { useState, useEffect } from 'react';
import { SignalModel } from '../../api/tradingSignals';
import { Card, CardHeader, CardBody } from '../Card';
import { Table, TableHeader, TableRow, TableCell } from '../Table';
import { formatDateTime, formatPercent } from '../../utils/formatters';
import { Pagination } from '../Pagination';

interface SignalHistoryProps {
  signals: SignalModel[];
  className?: string;
}

/**
 * Signal History Component
 * 
 * Displays a paginated history of trading signals
 */
const SignalHistory: React.FC<SignalHistoryProps> = ({ signals, className = '' }) => {
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(10);
  const [filteredSignals, setFilteredSignals] = useState<SignalModel[]>([]);
  
  // Calculate total pages
  const totalPages = Math.ceil(filteredSignals.length / pageSize);
  
  // Get current signals for the page
  const currentSignals = filteredSignals.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );
  
  // Update filtered signals when signals prop changes
  useEffect(() => {
    setFilteredSignals([...signals].sort((a, b) => {
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    }));
    setCurrentPage(1);
  }, [signals]);
  
  // Function to get signal color based on direction
  const getSignalColor = (direction: string): string => {
    if (direction.includes('BUY') || direction.includes('STRONG_BUY')) {
      return 'text-green-600';
    } else if (direction.includes('SELL') || direction.includes('STRONG_SELL')) {
      return 'text-red-600';
    }
    return 'text-gray-600';
  };
  
  return (
    <Card className={className}>
      <CardHeader>
        <h2 className="text-xl font-semibold">Signal History</h2>
        <div className="text-sm text-gray-500">
          Showing {filteredSignals.length} signals
        </div>
      </CardHeader>
      <CardBody>
        {filteredSignals.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No signal history available.
          </div>
        ) : (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableCell as="th">Timestamp</TableCell>
                  <TableCell as="th">Symbol</TableCell>
                  <TableCell as="th">Signal Type</TableCell>
                  <TableCell as="th">Direction</TableCell>
                  <TableCell as="th">Strength</TableCell>
                  <TableCell as="th">Confidence</TableCell>
                  <TableCell as="th">Source</TableCell>
                </TableRow>
              </TableHeader>
              <tbody>
                {currentSignals.map((signal, index) => (
                  <TableRow key={index}>
                    <TableCell>{formatDateTime(signal.timestamp)}</TableCell>
                    <TableCell>{signal.symbol}</TableCell>
                    <TableCell>{signal.signal_type}</TableCell>
                    <TableCell className={getSignalColor(signal.direction)}>
                      {signal.direction}
                    </TableCell>
                    <TableCell>{formatPercent(signal.strength)}</TableCell>
                    <TableCell>{formatPercent(signal.confidence)}</TableCell>
                    <TableCell>{signal.source}</TableCell>
                  </TableRow>
                ))}
              </tbody>
            </Table>
            
            {totalPages > 1 && (
              <div className="mt-4 flex justify-center">
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={setCurrentPage}
                />
              </div>
            )}
          </>
        )}
      </CardBody>
    </Card>
  );
};

export default SignalHistory;
