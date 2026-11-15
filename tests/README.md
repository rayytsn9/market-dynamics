# Test status

Last run: `pytest` from the project root, 10 Nov 2026. Result: **43 passed, 59 xfailed**.

The only test module is `tests/test_ta_lib.py`. Fixtures and parametrization live in `tests/conftest.py`. There are no fetcher, model, or database tests.

`test_indicator` is one test per entry in `StockConfig.TALIB_MAP`. A `None` value is marked `xfail` ("not yet implemented"). A name that `talib.abstract` does not provide is also `xfail`. The lists below are that split.

## Working

| Test | Marker | What it checks |
|---|---|---|
| `TestTa.test_initialize_ta` | `ta_fast` | `instantiate_factory` accepts the map with `None` entries removed (41 names). Each blueprint value is a string or a tuple, and the blueprint length matches that map. The `is_valid=False` branch is not a separate test. |
| `TestTa.test_workflow` | `pipeline` | Calls `compute` for every non-`None` map entry on the synthetic OHLCV fixture and prints the results. There is no assertion. It passes when none of those calls raise. |
| `TestTa.test_indicator` | `ta_pipeline` | For each mapped name, `compute` on the synthetic 100-row OHLCV frame returns a value other than `None`. |

The 41 `test_indicator` cases that pass:

| Indicator | TA-Lib |
|---|---|
| Accumulation Distribution | `AD` |
| ADX | `ADX` |
| Aroon Oscillator | `AROONOSC` |
| Average True Range | `ATR` |
| Bollinger Bands | `BBANDS` |
| Chaikin Oscillator | `ADOSC` |
| Chande Momentum Oscillator | `CMO` |
| Commodity Channel Index | `CCI` |
| Directional Movement | `DX` |
| Exponential Moving Average | `EMA` |
| Kaufman Adaptive Moving Average | `KAMA` |
| Linear Regression | `LINEARREG` |
| Linear Regression Indicator | `TSF` |
| MACD Histogram | `MACD` column `macdhist` |
| MACD Indicator | `MACD` column `macd` |
| MACD Percentage | `PPO` |
| Median Price | `MEDPRICE` |
| Momentum Indicator | `MOM` |
| Money Flow Index | `MFI` |
| Moving Average | `MA` |
| On Balance Volume | `OBV` |
| Parabolic SAR | `SAR` |
| Percentage Price Oscillator | `PPO` |
| Price Differential | `SUB` |
| Price Ratio | `DIV` |
| Rate of Change (Price) | `ROC` |
| Relative Strength Index (RSI) | `RSI` |
| Simple Moving Average | `SMA` |
| Standard Deviation Channels | `STDDEV` |
| Stochastic Oscillator | `STOCH` |
| Stochastic RSI | `STOCHRSI` |
| The Moving Average Oscillator | `APO` |
| Triple Exponential Moving Average | `TEMA` |
| TRIX Indicator | `TRIX` |
| True Range | `TRANGE` |
| Typical Price | `TYPPRICE` |
| Ultimate Oscillator | `ULTOSC` |
| Volatility | `STDDEV` |
| Weighted Close | `WCLPRICE` |
| Weighted Moving Average | `WMA` |
| Williams %R | `WILLR` |

OHLCV for these tests is the `sample_market_data` fixture (random `open`, `high`, `low`, `close`, `volume`). Nothing in the suite calls Yahoo Finance or CCXT.

## Not implemented

These 59 `test_indicator` cases are expected failures. `TALIB_MAP` stores `None` for each of them, so `compute` returns `None` and the assertion fails. Pytest records that as `xfail`, not as a regression.

ATR Bands, ATR Trailing Stops, Awesome Oscillator, Bollinger %B, Bollinger Band® Width, Candlestick Patterns, Chaikin Money Flow, Chaikin Volatility, Chandelier Exits, Choppiness Index, Compare Prices, Coppock Indicator, Detrended Price Oscillator, Displaced Moving Average, Donchian Channels, Ease of Movement, Elder Ray Index, Fibonacci Extensions, Fibonacci Retracements, Fibonacci Weighted Moving Average, Force Index, Heikin-Ashi Candlesticks, HLC3, Holt-Winters Moving Average, Hull Moving Average, Ichimoku Cloud, Inverted Axis, Keltner Channels, KST Indicator, Mass Index, Moving Average Filters, Moving Average High/Low/Open, Multiple Moving Averages, Negative Volume, Percentage Bands, Percentage Trailing Stops, Percentage Volume Oscillator, Pivot Points, Positive Volume, Price Comparison, Price Distance, Price Envelope, Price Volume Trend, Rainbow 3D Moving Averages, Rate of Change (Volume), Relative Strength (Compare), Safezone Indicator, Symmetric Weighted Moving Average, Trend Lines, True Strength Index, Ulcer Index, Vertical Horizontal Filter (VHF), Volatility Ratio, Volatility Stops, Volume, Volume Oscillator, Volume Weighted Average Price, Wilder Moving Average, Williams Accumulation Distribution.

## Registered markers with no tests

`pytest.ini` still defines `fast`, `slow`, and `ta_slow`. No test uses them.
