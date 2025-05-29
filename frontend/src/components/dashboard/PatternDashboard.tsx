import React, { useState, useEffect } from 'react';
import { 
  PatternDetectionResult, 
  PatternType, 
  PATTERN_COLORS 
} from '../../types/patterns';
// Import UI components directly from their files
import { Card, CardHeader, CardBody } from '../ui/Card';
import { Button } from '../ui/Button';
import { Tooltip } from '../ui/Tooltip';
import { Switch } from '../ui/Switch';
import { Badge } from '../ui/Badge';

// Import table components from index
import { 
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHeaderCell,
  TableCell 
} from '../ui/index';

interface PatternDashboardProps {
  patterns: PatternDetectionResult[];
  symbol: string;
  onPatternVisibilityChange?: (visiblePatternTypes: Record<PatternType, boolean>) => void;
  onShowPatternsChange?: (showPatterns: boolean) => void;
  showPatterns?: boolean;
  className?: string;
}

const PatternDashboard: React.FC<PatternDashboardProps> = ({
  patterns = [],
  symbol,
  onPatternVisibilityChange,
  onShowPatternsChange,
  showPatterns = true,
  className = '',
}) => {
  // Filter patterns to only show those for the current symbol
  const symbolPatterns = patterns.filter(p => p.symbol === symbol);
  
  // Track which pattern types are visible
  const [patternVisibility, setPatternVisibility] = useState<Record<PatternType, boolean>>(
    Object.values(PatternType).reduce((acc, type) => ({ ...acc, [type]: true }), {} as Record<PatternType, boolean>)
  );
  
  // Confidence threshold filter
  const [confidenceThreshold, setConfidenceThreshold] = useState(30); // Default 30%
  
  // Count of patterns by type
  const patternCounts = symbolPatterns.reduce((acc, pattern) => {
    if (!acc[pattern.pattern_type]) {
      acc[pattern.pattern_type] = 0;
    }
    acc[pattern.pattern_type]++;
    return acc;
  }, {} as Record<PatternType, number>);
  
  // Handle pattern type visibility toggle
  const handlePatternTypeToggle = (patternType: PatternType) => {
    const updatedVisibility = {
      ...patternVisibility,
      [patternType]: !patternVisibility[patternType]
    };
    setPatternVisibility(updatedVisibility);
    if (onPatternVisibilityChange) {
      onPatternVisibilityChange(updatedVisibility);
    }
  };
  
  // Handle show all patterns toggle
  const handleShowPatternsToggle = () => {
    if (onShowPatternsChange) {
      onShowPatternsChange(!showPatterns);
    }
  };
  
  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Get pattern count that passes confidence threshold
  const getFilteredPatternCount = () => {
    return symbolPatterns.filter(p => p.confidence >= confidenceThreshold).length;
  };
  
  return (
    <Card className={`bg-slate-800 shadow-md ${className}`}>
      <CardHeader className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-white">Technical Patterns</h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-300">Show Patterns</span>
          <Switch
            checked={showPatterns}
            onChange={handleShowPatternsToggle}
            size="sm"
          />
        </div>
      </CardHeader>
      <CardBody>
        {symbolPatterns.length === 0 ? (
          <div className="text-center py-4 text-gray-400">
            No patterns detected for {symbol}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Pattern Type Filters */}
            <div className="flex flex-wrap gap-2 mb-2">
              {Object.entries(patternCounts).map(([type, count]) => (
                <div
                  key={type}
                  style={{ 
                    backgroundColor: patternVisibility[type as PatternType] 
                      ? PATTERN_COLORS[type as PatternType] 
                      : 'rgba(75, 85, 99, 0.5)'
                  }}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium cursor-pointer flex items-center gap-1 text-white"
                  onClick={() => handlePatternTypeToggle(type as PatternType)}
                >
                  {type.replace('_', ' ')}
                  <span className="ml-1 text-xs bg-black/20 px-1 rounded">
                    {count}
                  </span>
                </div>
              ))}
            </div>
            
            {/* Confidence Filter */}
            <div className="flex items-center mb-4">
              <span className="text-sm text-gray-300 mr-2">Min Confidence:</span>
              <input
                type="range"
                min="0"
                max="100"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
                className="w-32"
              />
              <span className="text-sm text-gray-300 ml-2">{confidenceThreshold}%</span>
            </div>
            
            {/* Pattern Table */}
            <div className="overflow-x-auto">
              <Table className="w-full text-sm">
                <TableHeader>
                  <TableRow className="bg-slate-700">
                    <TableHeaderCell>Pattern</TableHeaderCell>
                    <TableHeaderCell>Time</TableHeaderCell>
                    <TableHeaderCell className="text-right">Confidence</TableHeaderCell>
                    <TableHeaderCell className="text-right">Target</TableHeaderCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {symbolPatterns
                    .filter(pattern => pattern.confidence >= confidenceThreshold)
                    .sort((a, b) => b.confidence - a.confidence)
                    .map((pattern, idx) => (
                      <TableRow 
                        key={idx} 
                        className={patternVisibility[pattern.pattern_type] ? '' : 'opacity-50'}
                      >
                        <TableCell className="px-2 py-2">
                          <div className="flex items-center">
                            <div 
                              className="w-3 h-3 rounded-full mr-2" 
                              style={{ backgroundColor: PATTERN_COLORS[pattern.pattern_type] }}
                            ></div>
                            <Tooltip content={pattern.pattern_type.replace('_', ' ')}>
                              <span>{pattern.pattern_type.split('_').map(word => word[0]).join('')}</span>
                            </Tooltip>
                          </div>
                        </TableCell>
                        <TableCell className="px-2 py-2 text-xs">
                          {formatTimestamp(pattern.end_time)}
                        </TableCell>
                        <TableCell className="px-2 py-2 text-right">
                          <Badge
                            variant={pattern.confidence >= 70 ? 'success' : 
                              pattern.confidence >= 50 ? 'warning' : 'danger'}
                          >
                            {Math.round(pattern.confidence)}%
                          </Badge>
                        </TableCell>
                        <TableCell className="px-2 py-2 text-right">
                          {pattern.target_price 
                            ? `$${pattern.target_price.toFixed(2)}` 
                            : '-'}
                        </TableCell>
                      </TableRow>
                    ))
                  }
                </TableBody>
              </Table>
            </div>
            
            <div className="text-xs text-gray-400 text-right mt-2">
              Showing {getFilteredPatternCount()} of {symbolPatterns.length} patterns
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  );
};

export default PatternDashboard;
