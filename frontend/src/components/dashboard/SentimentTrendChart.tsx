import { ChartOptions, createChart, SeriesOptions } from 'lightweight-charts';
import React, { useEffect, useRef, useState } from 'react';
import { getMockHistoricalSentiment } from '../../api/mockData/mockHistoricalSentiment'; // Assuming this mock data function exists
import { useDataSource } from '../../context/DataSourceContext';
// import { sentimentApi } from '../../api/sentiment'; // Assuming a historical sentiment API exists

interface SentimentTrendChartProps {
    symbol: string;
    timeframe?: string; // e.g., '1D', '1W', '1M', '1Y'
}

const SentimentTrendChart: React.FC<SentimentTrendChartProps> = ({ symbol, timeframe = '1M' }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<any | null>(null);
    const sentimentSeriesRef = useRef<any | null>(null);

    const { dataSource } = useDataSource();
    const [sentimentData, setSentimentData] = useState<any[] | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let isMounted = true;
        setIsLoading(true);
        setError(null);
        const fetchHistoricalSentiment = async () => {
            try {
                const data = dataSource === 'mock'
                    ? await getMockHistoricalSentiment(symbol, timeframe)
                    // : await sentimentApi.getHistoricalSentiment(symbol, timeframe); // Assuming this API exists
                    : []; // Fallback for non-mock data until API is implemented

                if (isMounted) {
                    // Format data for lightweight-charts
                    const formattedData: any[] = data.map((item: { timestamp: string; score: number }) => ({
                        time: Math.floor(new Date(item.timestamp).getTime() / 1000),
                        value: item.score, // Assuming the data has a 'score' field
                    }));
                    setSentimentData(formattedData);
                }
            } catch (e: any) {
                if (isMounted) {
                    console.error("Error fetching historical sentiment:", e);
                    setError("Failed to load sentiment data.");
                    setSentimentData(null);
                }
            } finally {
                if (isMounted) {
                    setIsLoading(false);
                }
            }
        };
        fetchHistoricalSentiment();
        return () => { isMounted = false; };
    }, [dataSource, symbol, timeframe]);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Clear previous chart
        chartContainerRef.current.innerHTML = '';

        const chartOptions: ChartOptions = {
            width: chartContainerRef.current.clientWidth,
            height: 200, // Adjust height as needed
            layout: {
                backgroundColor: '#ffffff',
                textColor: '#333',
            },
            grid: {
                vertLines: { color: '#e0e0eb' },
                horzLines: { color: '#e0e0eb' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
            // rightPriceScale: { // This option seems to be causing a TS error, removing for now
            //     visible: true,
            //     mode: 0, // Normal mode
            //     autoScale: true,
            //     invertScale: false,
            //     alignLabels: true,
            //     borderVisible: false,
            //     borderColor: '#555ffd',
            //     entireTextOnly: false,
            //     drawTicks: true,
            //     // Adjust price scale to fit sentiment scores (-1 to 1)
            //     // This might need adjustment based on actual sentiment score range
            //     scaleMargins: {
            //         top: 0.2,
            //         bottom: 0.2,
            //     },
            // },
        };

        const chart = createChart(chartContainerRef.current, chartOptions);

        chartRef.current = chart;

        const sentimentSeriesOptions: SeriesOptions = {
            color: '#2962FF',
            lineWidth: 2,
        };

        const sentimentSeries = chart.addLineSeries(sentimentSeriesOptions);
        sentimentSeriesRef.current = sentimentSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
                sentimentSeriesRef.current = null;
            }
        };
    }, [symbol, timeframe]); // Re-create chart if symbol or timeframe changes

    useEffect(() => {
        if (sentimentSeriesRef.current && sentimentData) {
            sentimentSeriesRef.current.setData(sentimentData);
            if (chartRef.current) {
                chartRef.current.timeScale().fitContent();
            }
        }
    }, [sentimentData]);

    if (isLoading) {
        return (
            <div className="dashboard-widget col-span-1">
                <h2 className="text-lg font-semibold mb-3">Sentiment Trend ({symbol})</h2>
                <div className="animate-pulse space-y-2">
                    <div className="h-40 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="dashboard-widget col-span-1">
                <h2 className="text-lg font-semibold mb-3">Sentiment Trend ({symbol})</h2>
                <div className="text-red-500 dark:text-red-400 text-center py-8 text-base font-medium">
                    {error}
                </div>
            </div>
        );
    }

    if (!sentimentData || sentimentData.length === 0) {
        return (
            <div className="dashboard-widget col-span-1">
                <h2 className="text-lg font-semibold mb-3">Sentiment Trend ({symbol})</h2>
                <div className="text-gray-500 dark:text-gray-400 text-center py-8 text-base font-medium">
                    No sentiment trend data available for {symbol}
                </div>
            </div>
        );
    }


    return (
        <div className="dashboard-widget col-span-1">
            <h2 className="text-lg font-semibold mb-3">Sentiment Trend ({symbol})</h2>
            <div ref={chartContainerRef} className="w-full h-[200px]" /> {/* Adjust height */}
        </div>
    );
};

export default SentimentTrendChart;
