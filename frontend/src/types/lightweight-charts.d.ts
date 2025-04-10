declare module 'lightweight-charts' {
  export interface ChartOptions {
    width?: number;
    height?: number;
    layout?: {
      backgroundColor?: string;
      textColor?: string;
    };
    grid?: {
      vertLines?: {
        color?: string;
      };
      horzLines?: {
        color?: string;
      };
    };
    timeScale?: {
      timeVisible?: boolean;
      secondsVisible?: boolean;
    };
  }

  export interface SeriesOptions {
    color?: string;
    lineWidth?: number;
    lineStyle?: number;
    title?: string;
    priceScaleId?: string;
    scaleMargins?: {
      top?: number;
      bottom?: number;
    };
    priceFormat?: {
      type?: string;
      precision?: number;
      minMove?: number;
    };
    upColor?: string;
    downColor?: string;
    borderVisible?: boolean;
    wickUpColor?: string;
    wickDownColor?: string;
  }

  export interface Chart {
    applyOptions(options: ChartOptions): void;
    resize(width: number, height: number): void;
    timeScale(): any;
    addCandlestickSeries(options?: SeriesOptions): Series;
    addLineSeries(options?: SeriesOptions): Series;
    addHistogramSeries(options?: SeriesOptions): Series;
    addAreaSeries(options?: SeriesOptions): Series;
    addBarSeries(options?: SeriesOptions): Series;
    removeSeries(series: Series): void;
  }

  export interface Series {
    setData(data: any[]): void;
    update(bar: any): void;
    setMarkers(markers: any[]): void;
    applyOptions(options: SeriesOptions): void;
  }

  export function createChart(container: HTMLElement, options?: ChartOptions): Chart;
}
