/**
 * Technical Analysis Pattern Types
 * 
 * These types define the data structure for technical analysis patterns
 * detected by the AI Trading Agent's pattern detection system.
 */

export enum PatternType {
  SUPPORT = 'SUPPORT',
  RESISTANCE = 'RESISTANCE',
  TRENDLINE_ASCENDING = 'TRENDLINE_ASCENDING',
  TRENDLINE_DESCENDING = 'TRENDLINE_DESCENDING',
  HEAD_AND_SHOULDERS = 'HEAD_AND_SHOULDERS',
  INVERSE_HEAD_AND_SHOULDERS = 'INVERSE_HEAD_AND_SHOULDERS',
  DOUBLE_TOP = 'DOUBLE_TOP',
  DOUBLE_BOTTOM = 'DOUBLE_BOTTOM',
  TRIANGLE_ASCENDING = 'TRIANGLE_ASCENDING',
  TRIANGLE_DESCENDING = 'TRIANGLE_DESCENDING',
  TRIANGLE_SYMMETRICAL = 'TRIANGLE_SYMMETRICAL',
  CUP_AND_HANDLE = 'CUP_AND_HANDLE',
  WEDGE_RISING = 'WEDGE_RISING',
  WEDGE_FALLING = 'WEDGE_FALLING',
  FLAG_BEARISH = 'FLAG_BEARISH',
  FLAG_BULLISH = 'FLAG_BULLISH'
}

export interface PatternDetectionResult {
  pattern_type: PatternType;
  symbol: string;
  confidence: number;
  start_time: string;
  end_time: string;
  target_price?: number;
  price_level?: number;
  additional_info?: Record<string, any>;
}

export interface PatternVisualizationData {
  id: string;
  patternType: PatternType;
  points: {
    x: number; // Time value
    y: number; // Price value
  }[];
  color: string;
  lineStyle: number; // 0 = solid, 1 = dotted, 2 = dashed
  isVisible: boolean;
  confidence: number;
  targetPrice?: number;
}

// Pattern color scheme for visualization
export const PATTERN_COLORS = {
  [PatternType.SUPPORT]: 'rgba(0, 128, 0, 0.7)', // Green
  [PatternType.RESISTANCE]: 'rgba(220, 0, 0, 0.7)', // Red
  [PatternType.TRENDLINE_ASCENDING]: 'rgba(0, 180, 0, 0.7)', // Bright Green
  [PatternType.TRENDLINE_DESCENDING]: 'rgba(180, 0, 0, 0.7)', // Bright Red
  [PatternType.HEAD_AND_SHOULDERS]: 'rgba(255, 165, 0, 0.7)', // Orange
  [PatternType.INVERSE_HEAD_AND_SHOULDERS]: 'rgba(255, 140, 0, 0.7)', // Dark Orange
  [PatternType.DOUBLE_TOP]: 'rgba(255, 0, 255, 0.7)', // Magenta
  [PatternType.DOUBLE_BOTTOM]: 'rgba(0, 255, 255, 0.7)', // Cyan
  [PatternType.TRIANGLE_ASCENDING]: 'rgba(128, 0, 128, 0.7)', // Purple
  [PatternType.TRIANGLE_DESCENDING]: 'rgba(0, 0, 128, 0.7)', // Navy
  [PatternType.TRIANGLE_SYMMETRICAL]: 'rgba(128, 128, 0, 0.7)', // Olive
  [PatternType.CUP_AND_HANDLE]: 'rgba(75, 0, 130, 0.7)', // Indigo
  [PatternType.WEDGE_RISING]: 'rgba(220, 20, 60, 0.7)', // Crimson
  [PatternType.WEDGE_FALLING]: 'rgba(34, 139, 34, 0.7)', // Forest Green
  [PatternType.FLAG_BEARISH]: 'rgba(178, 34, 34, 0.7)', // Firebrick
  [PatternType.FLAG_BULLISH]: 'rgba(46, 139, 87, 0.7)' // Sea Green
};

// Map pattern types to line styles
export const PATTERN_LINE_STYLES = {
  [PatternType.SUPPORT]: 0, // Solid
  [PatternType.RESISTANCE]: 0, // Solid
  [PatternType.TRENDLINE_ASCENDING]: 0, // Solid
  [PatternType.TRENDLINE_DESCENDING]: 0, // Solid
  [PatternType.HEAD_AND_SHOULDERS]: 1, // Dotted
  [PatternType.INVERSE_HEAD_AND_SHOULDERS]: 1, // Dotted
  [PatternType.DOUBLE_TOP]: 1, // Dotted
  [PatternType.DOUBLE_BOTTOM]: 1, // Dotted
  [PatternType.TRIANGLE_ASCENDING]: 2, // Dashed
  [PatternType.TRIANGLE_DESCENDING]: 2, // Dashed
  [PatternType.TRIANGLE_SYMMETRICAL]: 2, // Dashed
  [PatternType.CUP_AND_HANDLE]: 1, // Dotted
  [PatternType.WEDGE_RISING]: 2, // Dashed
  [PatternType.WEDGE_FALLING]: 2, // Dashed
  [PatternType.FLAG_BEARISH]: 1, // Dotted
  [PatternType.FLAG_BULLISH]: 1 // Dotted
};
