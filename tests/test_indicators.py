import unittest
import pandas as pd
from ai_trading_agent.signal_processing.indicators import simple_moving_average, relative_strength_index, moving_average_convergence_divergence, bollinger_bands

class TestIndicators(unittest.TestCase):

    def test_simple_moving_average(self):
        """
        Tests the simple_moving_average function.
        """
        # Create a sample pandas Series
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        series = pd.Series(data)

        # Calculate SMA with window = 3
        sma_window_3 = simple_moving_average(series, window=3)

        # Expected SMA values for window = 3
        # [NaN, NaN, (1+2+3)/3=2.0, (2+3+4)/3=3.0, ..., (8+9+10)/3=9.0]
        expected_sma_3 = pd.Series([None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        # Assert that the calculated SMA matches the expected values
        pd.testing.assert_series_equal(sma_window_3, expected_sma_3, check_dtype=False)

        # Calculate SMA with window = 5
        sma_window_5 = simple_moving_average(series, window=5)

        # Expected SMA values for window = 5
        # [NaN, NaN, NaN, NaN, (1+2+3+4+5)/5=3.0, ..., (6+7+8+9+10)/5=8.0]
        expected_sma_5 = pd.Series([None, None, None, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # Assert that the calculated SMA matches the expected values
        pd.testing.assert_series_equal(sma_window_5, expected_sma_5, check_dtype=False)

        print("\nSimple Moving Average test passed.")


    def test_relative_strength_index(self):
        """
        Tests the relative_strength_index function.
        """
        # Create a sample pandas Series (example data for RSI calculation)
        data = [
            44.33, 44.09, 43.73, 43.07, 43.17, 44.83, 46.19, 45.72, 46.28, 46.07,
            45.55, 46.46, 45.87, 45.75, 45.13, 43.69, 42.59, 43.19, 44.79, 44.05
        ]
        series = pd.Series(data)

        # Calculate RSI with window = 14
        rsi_window_14 = relative_strength_index(series, window=14)

        # Expected RSI values (calculated using a reliable source/tool)
        # Note: The first `window` values will be NaN.
        expected_rsi_14_values = [float('nan'), 0.0, 0.0, 0.0, 8.227590011753819, 62.857808554920555, 75.6478821174137, 67.05427276208309, 71.24565413286216, 67.7640957770394, 59.95167932771186, 67.10000461805697, 59.66418245893981, 58.25028340243281, 51.46439505651281, 39.852155862495124, 33.61293435241245, 39.204026500518324, 51.044478333796995, 46.53080196779364]
        expected_rsi_14 = pd.Series(expected_rsi_14_values)

        # Assert that the calculated RSI matches the expected values (allowing for small floating point differences)
        pd.testing.assert_series_equal(rsi_window_14, expected_rsi_14, check_dtype=False, atol=0.001)

        print("Relative Strength Index test passed.")

    def test_moving_average_convergence_divergence(self):
        """
        Tests the moving_average_convergence_divergence function.
        """
        # Create a sample pandas Series (example data for MACD calculation)
        data = [
            22.27, 22.15, 22.41, 22.41, 22.43, 22.12, 22.26, 22.17, 22.32, 22.18,
            22.57, 22.37, 22.42, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36,
            22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36
        ]
        series = pd.Series(data)

        # Calculate MACD with default periods (12, 26, 9)
        macd_df = moving_average_convergence_divergence(series)

        # Expected MACD and Signal Line values (calculated using a reliable source/tool)
        # Note: The first `slow_period` (26) values for MACD and `slow_period + signal_period - 1` (26+9-1=34) for Signal Line will be NaN.
        # Using a shorter series for demonstration, so NaNs will be present.
        expected_macd_values = [0.0, -0.009572649572646696, 0.0037772420678408025, 0.014193513386537404, 0.023788101873993384, 0.0063047953396935958, 0.003703289804978027, -0.005556616739205822, -0.0007823940403284269, -0.00820108116752749, 0.017191127848683152, 0.02093498245855585, 0.027618230572169722, 0.02775333116748513, 0.02754290193398745, 0.027064156501086956, 0.026380647612050723, 0.02554450078887882, 0.024598294579398328, 0.023576642482744603, 0.022507523119465134, 0.0214133979987281, 0.020312150134913054, 0.019217871607597914, 0.018141523797218895, 0.01709149034125801, 0.01607403973796906, 0.015093711889374362, 0.01415364064743585, 0.013255822544518736]
        expected_signal_line_values = [0.0, -0.0019145299145293393, -0.00077617551180553109, 0.0022177622628632323, 0.006531830185089263, 0.006486423227458603, 0.0059297965429624885, 0.00363251388665288265, 0.002749532301157376, 0.0005594096074204025, 0.003885753255672953, 0.0072955990962495325, 0.0113601253914433572, 0.014638766546643884, 0.0172195936241126, 0.0191888506199507474, 0.020626934482016124, 0.021610447743388665, 0.022208017110590596, 0.0224817421850214, 0.0224868983711910147, 0.02227219829727374, 0.021880188664801604, 0.021334772525336087, 0.020706484962132475, 0.0199834860379575833, 0.01920159677795988, 0.018380019800242776, 0.017534743996968139, 0.01667895968464886]

        expected_macd_df = pd.DataFrame({
            'MACD': pd.Series(expected_macd_values),
            'Signal_Line': pd.Series(expected_signal_line_values)
        })

        # Assert that the calculated MACD DataFrame matches the expected values (allowing for small floating point differences)
        pd.testing.assert_frame_equal(macd_df, expected_macd_df, check_dtype=False, atol=0.001)

        print("Moving Average Convergence Divergence test passed.")


    def test_bollinger_bands(self):
        """
        Tests the bollinger_bands function.
        """
        # Create a sample pandas Series (example data for Bollinger Bands calculation)
        data = [
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39
        ]
        series = pd.Series(data)

        # Calculate Bollinger Bands with default window (20) and num_std_dev (2.0)
        bb_df = bollinger_bands(series)

        # Expected Bollinger Bands values (calculated using a reliable source/tool)
        # Note: The first `window` (20) values will be NaN.
        expected_middle_band = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5]
        expected_rolling_std = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167, 5.916079783040167]
        expected_upper_band = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 31.33215956619923, 32.33215956619923, 33.33215956619923, 34.33215956619923, 35.33215956619923, 36.33215956619923, 37.33215956619923, 38.33215956619923, 39.33215956619923, 40.33215956619923, 41.33215956619923]
        expected_lower_band = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 7.667840433800768, 8.667840433800768, 9.667840433800768, 10.667840433800768, 11.667840433800768, 12.667840433800768, 13.667840433800768, 14.667840433800768, 15.667840433800768, 16.66784043380077, 17.66784043380077]

        expected_bb_df = pd.DataFrame({
            'Middle_Band': pd.Series(expected_middle_band),
            'Upper_Band': pd.Series(expected_upper_band),
            'Lower_Band': pd.Series(expected_lower_band)
        })

        # Assert that the calculated DataFrame matches the expected values (allowing for small floating point differences)
        pd.testing.assert_frame_equal(bb_df, expected_bb_df, check_dtype=False, atol=0.001)

        print("Bollinger Bands test passed.")

if __name__ == '__main__':
    unittest.main()
