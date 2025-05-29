import React from 'react';
import { PatternType, PATTERN_COLORS } from '../../types/patterns';
// Import UI components directly from their files
import { Card, CardHeader, CardBody } from '../ui/Card';
import { Switch } from '../ui/Switch';
import { Tooltip } from '../ui/Tooltip';

interface PatternVisualizationControlsProps {
  patternVisibility: Record<PatternType, boolean>;
  onPatternVisibilityChange: (visibility: Record<PatternType, boolean>) => void;
  displayedPatterns: PatternType[];
  showPatterns: boolean;
  onShowPatternsChange: (show: boolean) => void;
  patternCounts?: Record<PatternType, number>;
  className?: string;
}

const PatternVisualizationControls: React.FC<PatternVisualizationControlsProps> = ({
  patternVisibility,
  onPatternVisibilityChange,
  displayedPatterns,
  showPatterns,
  onShowPatternsChange,
  patternCounts = {},
  className = '',
}) => {
  // Toggle all patterns on or off
  const handleToggleAll = (value: boolean) => {
    const updatedVisibility = { ...patternVisibility };
    Object.keys(updatedVisibility).forEach(key => {
      updatedVisibility[key as PatternType] = value;
    });
    onPatternVisibilityChange(updatedVisibility);
  };

  // Toggle a specific pattern type
  const handleTogglePattern = (patternType: PatternType) => {
    const updatedVisibility = {
      ...patternVisibility,
      [patternType]: !patternVisibility[patternType]
    };
    onPatternVisibilityChange(updatedVisibility);
  };

  // Get nice display name from pattern type enum
  const getPatternDisplayName = (patternType: PatternType): string => {
    return patternType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  return (
    <Card className={`bg-slate-800 shadow-md ${className}`}>
      <CardHeader className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-white">Pattern Visualization</h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-300">Show Patterns</span>
          <Switch
            checked={showPatterns}
            onChange={() => onShowPatternsChange(!showPatterns)}
            size="sm"
          />
        </div>
      </CardHeader>
      <CardBody>
        {displayedPatterns.length === 0 ? (
          <div className="text-center py-2 text-gray-400">
            No patterns detected
          </div>
        ) : (
          <>
            <div className="flex justify-between mb-3">
              <button
                className="text-xs text-blue-400 hover:text-blue-300"
                onClick={() => handleToggleAll(true)}
              >
                Show All
              </button>
              <button
                className="text-xs text-blue-400 hover:text-blue-300"
                onClick={() => handleToggleAll(false)}
              >
                Hide All
              </button>
            </div>
            <div className="grid grid-cols-1 gap-2">
              {displayedPatterns.map(patternType => (
                <div 
                  key={patternType}
                  className="flex items-center justify-between bg-slate-700 rounded p-2"
                >
                  <div className="flex items-center">
                    <div 
                      className="w-3 h-3 rounded-full mr-2" 
                      style={{ backgroundColor: PATTERN_COLORS[patternType] }}
                    ></div>
                    <Tooltip content={getPatternDisplayName(patternType)}>
                      <span className="text-sm text-white">
                        {getPatternDisplayName(patternType)}
                      </span>
                    </Tooltip>
                    {patternCounts[patternType] && (
                      <span className="ml-2 bg-slate-600 text-white text-xs px-2 py-0.5 rounded-full">
                        {patternCounts[patternType]}
                      </span>
                    )}
                  </div>
                  <Switch
                    checked={patternVisibility[patternType]}
                    onChange={() => handleTogglePattern(patternType)}
                    size="sm"
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
};

export default PatternVisualizationControls;
