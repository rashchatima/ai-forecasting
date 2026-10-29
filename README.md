# Seasonal time-series forecasting: SARIMA vs ARIMA

Forecasting of monthly sales with a yearly seasonal pattern, comparing a seasonal **SARIMA(1,1,1)×(1,1,1,12)**
model with a non-seasonal **ARIMA(2,1,2)** (Python, `statsmodels`).

## What it does

1. Generates 96 months of synthetic sales: linear trend + yearly seasonality with a December peak + noise.
2. Checks stationarity with the augmented Dickey–Fuller test before and after differencing.
3. Fits both models on the training part and forecasts the last 6, 12 or 24 months.
4. Measures accuracy with **MSE** and **MAE** and draws the forecast.

## Results (MAE, lower is better)

| Horizon | SARIMA | ARIMA |
|---|---|---|
| 6 months | **5.55** | 6.27 |
| 12 months | **5.11** | 11.01 |
| 24 months | **3.97** | 26.05 |

The seasonal model wins on every horizon, and the gap grows with the horizon: a model without seasonal
terms cannot reproduce the yearly cycle far ahead.

## Run

```bash
pip install -r requirements.txt
python forecasting.py                                  # interactive: choose model and horizon
python forecasting.py --model both --horizon 12 --plot forecast.png
python test_forecasting.py                             # 23 checks, writes test_report.html with plots
```

Input is validated: wrong menu values are asked again, end of input (Ctrl+Z / EOF) exits cleanly,
and wrong command-line arguments are rejected by `argparse`.

---

Author: **Tymofii Rashchenko**
